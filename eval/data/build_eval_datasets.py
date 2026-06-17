#!/usr/bin/env python3
"""
One-click evaluation dataset builder for ShapeLLM-Omni.

Downloads Objaverse meshes, fetches PointLLM GT, loads 3D-Alpaca from HuggingFace,
and builds JSON datasets for all three eval tasks:
  1. Understanding (3D captioning): PointLLM 200 + 3D-Alpaca caption subset
  2. Generation (text-to-3D): 3D-Alpaca text-to-3D prompts
  3. VQVAE Reconstruction: all downloaded meshes

Usage:
    python -m eval.data.build_eval_datasets --output_dir eval_data --mesh_cache_dir /data0/objaverse_meshes

    # Quick test with fewer samples
    python -m eval.data.build_eval_datasets --output_dir eval_data --mesh_cache_dir /data0/objaverse_meshes --max_alpaca_caption 100 --max_alpaca_gen 100

    # Skip downloads (if meshes are already cached)
    python -m eval.data.build_eval_datasets --output_dir eval_data --mesh_cache_dir /data0/objaverse_meshes --skip_download
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POINTLLM_GT_URLS = {
    200: "https://huggingface.co/datasets/RunsenXu/PointLLM/resolve/main/PointLLM_brief_description_val_200_GT.json",
    3000: "https://huggingface.co/datasets/RunsenXu/PointLLM/resolve/main/PointLLM_brief_description_val_3000_GT.json",
}

ALPACA_EDIT_DATA_URL = "https://hf-mirror.com/datasets/yejunliang23/3D-Alpaca/resolve/main/edit_data.json"

MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}

# Standard generation prompts used by ShapeLLM-Omni for text-to-3D
GENERATION_PROMPTS = [
    "A drone with four propellers and a central body.",
    "A stone axe with a handle.",
    "The titanic, aerial view.",
    "A 3D model of a small yellow and blue robot with wheels and two pots.",
    "A futuristic vehicle with a sleek design and multiple wheels.",
    "A car with four wheels and a roof.",
    "A wooden chair with four legs and armrests.",
    "A medieval castle with towers and walls.",
    "A rocket ship ready for launch.",
    "A small sailing boat on water.",
    "A vintage typewriter with round keys.",
    "A grand piano with an open lid.",
    "A mechanical clock with visible gears.",
    "A steam locomotive with smoke coming from the chimney.",
    "A lighthouse on a rocky cliff.",
    "A small house with a chimney and a fence.",
    "A military tank with a turret and tracks.",
    "A space station orbiting Earth.",
    "A submarine with a periscope.",
    "A vintage camera with a large lens.",
    "A windmill with four blades.",
    "A helicopter with rotors and tail fin.",
    "A microscope with adjustment knobs.",
    "A telescope on a tripod.",
    "A compass with a rotating needle.",
    "A viking ship with a dragon figurehead.",
    "A hot air balloon with a basket.",
    "A fire truck with a ladder.",
    "A bicycle with thin wheels.",
    "A motorcycle with a sidecar.",
    "A sports car with aerodynamic design.",
    "A classic pickup truck.",
    "A double-decker bus.",
    "A tractor with large rear wheels.",
    "A skateboard with colorful graphics.",
    "A snowboard with bindings.",
    "A surfboard with a pointed nose.",
    "A kayak with a paddle.",
    "A canoe for two people.",
    "A jet ski on water.",
    "An electric guitar with six strings.",
    "A violin with a bow.",
    "A drum set with cymbals.",
    "A trumpet with three valves.",
    "A flute with silver keys.",
    "A harp with golden frame.",
    "A saxophone with a curved body.",
    "A cello standing upright.",
    "A xylophone with colored bars.",
    "An accordion with bellows.",
    "A teddy bear with a bow tie.",
    "A rubber duck.",
    "A chess piece - the knight.",
    "A chess piece - the king.",
    "A globe on a stand.",
    "A magnifying glass with a wooden handle.",
    "A lantern with a candle inside.",
    "A treasure chest with gold coins.",
    "A crown with jewels.",
    "A shield with a coat of arms.",
    "A sword with an ornate handle.",
    "A bow and arrow.",
    "A catapult ready to fire.",
    "A cannon on wheels.",
    "A torch with flames.",
    "A wizard hat with stars.",
    "A magic wand with a glowing tip.",
    "A crystal ball on a stand.",
    "A potion bottle with bubbling liquid.",
    "A scroll with ancient writing.",
    "A mushroom with red cap and white spots.",
    "A cactus in a pot.",
    "A bonsai tree.",
    "A sunflower in full bloom.",
    "A rose with thorns.",
    "A palm tree on a beach.",
    "An oak tree with broad branches.",
    "A pine tree covered in snow.",
    "A bamboo stalk.",
    "A pumpkin with a carved face.",
    "A watermelon slice.",
    "An apple with a leaf.",
    "A banana bunch.",
    "A strawberry with seeds.",
    "A pineapple with a crown.",
    "A bunch of grapes.",
    "A cherry pair on a stem.",
    "A lemon cut in half.",
    "An orange with a peel.",
    "A birthday cake with candles.",
    "A cupcake with frosting.",
    "A donut with sprinkles.",
    "A slice of pizza.",
    "A hamburger with lettuce and tomato.",
    "A hot dog with mustard.",
    "A taco with fillings.",
    "A sushi roll.",
    "An ice cream cone with three scoops.",
    "A coffee cup with steam.",
    # --- Vehicles & Transport (extended) ---
    "A school bus with stop sign.",
    "A police car with sirens.",
    "An ambulance with a red cross.",
    "A garbage truck with a lift arm.",
    "A cement mixer truck.",
    "A forklift carrying a pallet.",
    "An airplane taking off.",
    "A fighter jet with missiles.",
    "A blimp floating in the sky.",
    "A speedboat cutting through waves.",
    "A cruise ship with multiple decks.",
    "A train with passenger cars.",
    "A monorail on elevated tracks.",
    "A cable car hanging from wires.",
    "A rickshaw with a seat.",
    "A horse-drawn carriage.",
    "A segway personal transporter.",
    "A hovercraft on water.",
    "A snowmobile on snow.",
    "A quad bike with large tires.",
    # --- Architecture & Buildings ---
    "A Gothic cathedral with stained glass.",
    "A Japanese pagoda with five tiers.",
    "A Greek temple with columns.",
    "A modern skyscraper with glass facade.",
    "A log cabin in the woods.",
    "A treehouse with a rope ladder.",
    "A bridge over a river.",
    "A water tower on stilts.",
    "A barn with a red roof.",
    "A greenhouse with glass panels.",
    "A gazebo in a garden.",
    "An igloo made of ice blocks.",
    "A pyramid with stone blocks.",
    "A colosseum with arched openings.",
    "A mosque with a dome and minaret.",
    "A church with a steeple.",
    "A factory with smokestacks.",
    "A stadium with seating rows.",
    "A phone booth, classic red.",
    "A mailbox on a post.",
    # --- Furniture & Household ---
    "A bookshelf filled with books.",
    "A round dining table with four chairs.",
    "A rocking chair on a porch.",
    "A desk lamp with adjustable arm.",
    "A standing floor lamp.",
    "A ceiling fan with lights.",
    "A wardrobe with double doors.",
    "A kitchen sink with faucet.",
    "A bathtub with clawed feet.",
    "A toilet with a tank.",
    "A washing machine with a round door.",
    "A refrigerator with two doors.",
    "A microwave oven.",
    "A toaster with two slots.",
    "A blender with a glass jar.",
    "A vacuum cleaner with a hose.",
    "An ironing board with an iron.",
    "A sewing machine with a needle.",
    "A grandfather clock with pendulum.",
    "A fireplace with a mantle.",
    # --- Electronics & Tech ---
    "A desktop computer with monitor and keyboard.",
    "A laptop computer, open.",
    "A smartphone with a large screen.",
    "A tablet device with a stylus.",
    "A game controller with joysticks.",
    "A pair of headphones.",
    "A speaker with a mesh grille.",
    "A projector on a ceiling mount.",
    "A router with antennas.",
    "A drone with a camera underneath.",
    "A VR headset.",
    "A smartwatch on a wrist.",
    "A 3D printer with a build plate.",
    "A CNC machine with cutting tool.",
    "A robot arm with joints.",
    "A satellite dish pointed upward.",
    "A walkie-talkie radio.",
    "A flashlight with a beam.",
    "A battery pack with cables.",
    "A solar panel on a roof.",
    # --- Animals ---
    "A cat sitting on a cushion.",
    "A dog with a wagging tail.",
    "A horse galloping.",
    "An elephant with large ears.",
    "A giraffe with a long neck.",
    "A lion with a mane.",
    "A penguin standing on ice.",
    "An eagle with spread wings.",
    "A dolphin jumping out of water.",
    "A shark with an open mouth.",
    "A whale breaching the surface.",
    "A turtle with a shell.",
    "A frog on a lily pad.",
    "A snake coiled up.",
    "A butterfly with colorful wings.",
    "A spider on a web.",
    "A bear standing on hind legs.",
    "A rabbit with long ears.",
    "An owl perched on a branch.",
    "A parrot on a perch.",
    # --- Tools & Equipment ---
    "A hammer with a wooden handle.",
    "A wrench with an adjustable jaw.",
    "A screwdriver with a flat head.",
    "A power drill with a bit.",
    "A chainsaw with a guide bar.",
    "A wheelbarrow filled with soil.",
    "A ladder leaning against a wall.",
    "A fire extinguisher.",
    "A toolbox with compartments.",
    "A paint roller with a handle.",
    "A shovel stuck in the ground.",
    "A pickaxe with a wooden handle.",
    "A crowbar.",
    "A level tool with a bubble.",
    "A tape measure extended.",
    "A clamp holding two boards.",
    "A vise grip on a workbench.",
    "A welding mask.",
    "A safety helmet, yellow.",
    "A pair of work gloves.",
    # --- Sports & Recreation ---
    "A basketball with orange texture.",
    "A soccer ball with black and white panels.",
    "A football with laces.",
    "A tennis racket with strings.",
    "A baseball bat and glove.",
    "A golf club and ball.",
    "A hockey stick and puck.",
    "A bowling pin.",
    "A dartboard with darts.",
    "A punching bag on a chain.",
    "A trampoline with safety net.",
    "A swing set with two swings.",
    "A slide for a playground.",
    "A seesaw.",
    "A sandbox with toys.",
    "A tent for camping.",
    "A sleeping bag rolled up.",
    "A fishing rod with a reel.",
    "A life jacket, orange.",
    "A pair of binoculars.",
    # --- Fantasy & Sci-Fi ---
    "A dragon breathing fire.",
    "A unicorn with a spiral horn.",
    "A phoenix with flaming wings.",
    "A spaceship with laser cannons.",
    "A mech warrior with weapons.",
    "An alien with large eyes.",
    "A flying saucer hovering.",
    "A portal with swirling energy.",
    "A magic staff with a crystal.",
    "A floating island with waterfalls.",
    "A time machine with dials.",
    "A robot butler serving drinks.",
    "A hoverboard glowing neon.",
    "A space helmet with visor.",
    "A laser rifle, futuristic.",
    "An energy shield generator.",
    "A holographic display table.",
    "A cyberpunk motorcycle.",
    "A steampunk airship.",
    "A medieval knight in full armor.",
    # --- Nature & Geography ---
    "A volcano with lava flowing.",
    "A mountain with snow peak.",
    "A waterfall cascading into a pool.",
    "A cave entrance with stalactites.",
    "A desert sand dune.",
    "A coral reef with fish.",
    "An iceberg floating in the sea.",
    "A canyon with layered rock.",
    "A river winding through a valley.",
    "A forest clearing with sunlight.",
    # --- Food & Kitchen (extended) ---
    "A croissant with flaky layers.",
    "A pretzel with salt.",
    "A bowl of ramen with chopsticks.",
    "A plate of spaghetti with sauce.",
    "A stack of pancakes with syrup.",
    "A loaf of bread on a board.",
    "A cheese wheel cut open.",
    "A wine bottle with a glass.",
    "A teapot with a lid.",
    "A jar of honey with a dipper.",
    # --- Miscellaneous ---
    "A pair of scissors.",
    "A padlock with a key.",
    "An umbrella, open.",
    "A suitcase with wheels.",
    "A backpack with straps.",
    "A gift box with a ribbon.",
    "A candle in a holder.",
    "A mirror with an ornate frame.",
    "A vase with flowers.",
    "A birdhouse on a pole.",
    "A mailbox, classic American style.",
    "A traffic light, red.",
    "A street lamp, vintage.",
    "A fire hydrant, red.",
    "A park bench under a tree.",
    "A fountain in a plaza.",
    "A statue on a pedestal.",
    "A clock tower.",
    "A weather vane on a roof.",
    "A sundial in a garden.",
    # --- Clothing & Accessories ---
    "A pair of running shoes.",
    "A top hat.",
    "A baseball cap.",
    "A pair of sunglasses.",
    "A wristwatch with a leather strap.",
    "A necktie with stripes.",
    "A handbag with a clasp.",
    "A belt with a buckle.",
    "A pair of boots.",
    "A scarf wrapped around a mannequin.",
    # --- Medical & Science ---
    "A stethoscope.",
    "A syringe with a needle.",
    "A pill bottle with tablets.",
    "A DNA double helix model.",
    "A molecule model, ball and stick.",
    "An atom model with orbiting electrons.",
    "A beaker with liquid.",
    "A test tube rack with tubes.",
    "A Bunsen burner with flame.",
    "A petri dish with culture.",
    # --- Office & School ---
    "A pencil sharpener.",
    "A stapler.",
    "A paper clip, large.",
    "A rubber stamp.",
    "A filing cabinet with drawers.",
    "A whiteboard with markers.",
    "A calculator.",
    "A hole puncher.",
    "A notebook with spiral binding.",
    "A pen holder with pens.",
    # --- Space & Astronomy ---
    "The planet Saturn with rings.",
    "The Earth from space.",
    "The Moon with craters.",
    "A Mars rover on rocky terrain.",
    "A space telescope in orbit.",
    "An asteroid with irregular shape.",
    "A comet with a glowing tail.",
    "A space shuttle on launch pad.",
    "An astronaut in a spacesuit.",
    "A lunar lander with legs.",
    # --- Marine & Underwater ---
    "An anchor with a chain.",
    "A ship's wheel.",
    "A pirate flag, skull and crossbones.",
    "A treasure map, rolled.",
    "A diving helmet, old style.",
    "A buoy floating on water.",
    "A fishing net.",
    "A seashell, conch.",
    "A starfish.",
    "A seahorse.",
    # --- Winter & Holiday ---
    "A snowman with a top hat.",
    "A Christmas tree with ornaments.",
    "A gingerbread house.",
    "A sled with runners.",
    "Ice skates, a pair.",
    "A menorah with candles.",
    "A jack-o-lantern glowing.",
    "A Easter egg with patterns.",
    "A firework rocket.",
    "A party hat with streamers.",
    # --- Industrial & Mechanical ---
    "A gear wheel with teeth.",
    "A pulley system with rope.",
    "A spring coil, metal.",
    "A piston and cylinder.",
    "A valve with a handle.",
    "A conveyor belt with boxes.",
    "A crane with a hook.",
    "An excavator with bucket.",
    "A bulldozer with blade.",
    "A steamroller on a road.",
    # --- Art & Crafts ---
    "An easel with a canvas.",
    "A paint palette with colors.",
    "A pottery wheel with clay.",
    "A sculpture bust.",
    "A mosaic tile pattern.",
    "A weaving loom.",
    "A knitting basket with yarn.",
    "An origami crane.",
    "A calligraphy brush and ink.",
    "A picture frame, ornate.",
]


# ---------------------------------------------------------------------------
# Step 1: Download PointLLM Ground Truth
# ---------------------------------------------------------------------------

def download_pointllm_gt(
    cache_dir: str, scale: int = 3000
) -> Tuple[str, List[Dict]]:
    """
    Download and parse PointLLM GT from HuggingFace.

    Args:
        cache_dir: Directory to cache the downloaded file.
        scale: 200 for the small benchmark, 3000 for the large one.
    """
    os.makedirs(cache_dir, exist_ok=True)
    url = POINTLLM_GT_URLS.get(scale, POINTLLM_GT_URLS[3000])
    filename = f"PointLLM_brief_description_val_{scale}_GT.json"
    gt_path = os.path.join(cache_dir, filename)

    if not os.path.exists(gt_path):
        print(f"[PointLLM] Downloading {scale}-sample GT from HuggingFace...")
        try:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx) as resp:
                data = resp.read()
            with open(gt_path, "wb") as f:
                f.write(data)
            print(f"[PointLLM] Saved to: {gt_path}")
        except Exception as e:
            print(f"[PointLLM] urllib failed ({e}), trying subprocess curl...")
            import subprocess
            subprocess.run(
                ["curl", "-sL", "--max-time", "60", url, "-o", gt_path],
                check=True,
            )
            print(f"[PointLLM] Saved via curl to: {gt_path}")
    else:
        print(f"[PointLLM] GT already cached at: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"[PointLLM] Loaded {len(gt_data)} GT samples")
    return gt_path, gt_data


# ---------------------------------------------------------------------------
# Step 2: Load 3D-Alpaca from HuggingFace
# ---------------------------------------------------------------------------

def load_3d_alpaca(
    cache_dir: str,
    max_caption: int = 1000,
    max_gen: int = 1200,
    max_editing: int = 200,
) -> Dict[str, List[Dict]]:
    """
    Load 3D-Alpaca dataset. Tries HuggingFace `datasets` first,
    falls back to direct download of edit_data.json via curl.

    Returns dict with keys: '3d_to_caption', 'text_to_3d', '3d_editing'
    """
    categorized: Dict[str, List[Dict]] = {
        "3d_to_caption": [],
        "text_to_3d": [],
        "3d_editing": [],
    }

    # Try loading via datasets library first
    raw_items = None
    try:
        from datasets import load_dataset
        print(f"[3D-Alpaca] Trying HuggingFace datasets library...")
        ds = load_dataset("yejunliang23/3D-Alpaca", trust_remote_code=True)
        split = "train"
        for s in ["test", "validation", "train"]:
            if s in ds:
                split = s
                break
        raw_items = list(ds[split])
        print(f"[3D-Alpaca] Loaded {len(raw_items)} items via datasets library")
    except Exception as e:
        print(f"[3D-Alpaca] datasets library failed: {e}")
        print(f"[3D-Alpaca] Falling back to direct download of edit_data.json...")

    # Fallback: download edit_data.json directly
    if raw_items is None:
        local_path = os.path.join(cache_dir, "edit_data.json")
        if not os.path.exists(local_path):
            print(f"[3D-Alpaca] Downloading edit_data.json (~1GB, this may take a while)...")
            import subprocess
            result = subprocess.run(
                ["curl", "-sL", "--max-time", "600", ALPACA_EDIT_DATA_URL, "-o", local_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0 or not os.path.exists(local_path):
                print(f"[3D-Alpaca] Download failed: {result.stderr}")
                print(f"[3D-Alpaca] Will use generated prompts for generation task instead")
                return categorized
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"[3D-Alpaca] Downloaded edit_data.json ({size_mb:.1f} MB)")

        print(f"[3D-Alpaca] Loading edit_data.json...")
        with open(local_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        print(f"[3D-Alpaca] Loaded {len(raw_items)} items")

    # Categorize by task type
    limits = {
        "3d_to_caption": max_caption,
        "text_to_3d": max_gen,
        "3d_editing": max_editing,
    }
    counts = defaultdict(int)

    for item in raw_items:
        # Support both conversation formats:
        # Format A (HF datasets): {"conversations": [{"from":"human","value":...}, ...]}
        # Format B (edit_data.json): {"messages": [{"role":"user","content":...}, ...]}
        conversations = item.get("conversations", [])
        messages = item.get("messages", [])

        human_msg = ""
        gpt_msg = ""

        if conversations:
            for conv in conversations:
                if conv.get("from") == "human":
                    human_msg = conv.get("value", "")
                elif conv.get("from") == "gpt":
                    gpt_msg = conv.get("value", "")
        elif messages:
            for msg in messages:
                if msg.get("role") == "user":
                    human_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    gpt_msg = msg.get("content", "")

        if not human_msg and not gpt_msg:
            continue

        has_mesh_input = "<mesh-start>" in human_msg or "<mesh" in human_msg
        has_mesh_output = "<mesh-start>" in gpt_msg or "<mesh" in gpt_msg

        if has_mesh_input and not has_mesh_output:
            task_type = "3d_to_caption"
        elif not has_mesh_input and has_mesh_output:
            task_type = "text_to_3d"
        elif has_mesh_input and has_mesh_output:
            task_type = "3d_editing"
        else:
            continue

        if counts[task_type] >= limits.get(task_type, 0):
            all_done = all(counts[t] >= limits[t] for t in limits)
            if all_done:
                break
            continue

        uid = item.get("object_id", item.get("uid", ""))

        categorized[task_type].append({
            "human_msg": human_msg,
            "gpt_msg": gpt_msg,
            "task_type": task_type,
            "uid": uid,
        })
        counts[task_type] += 1

    for task_type, items in categorized.items():
        print(f"[3D-Alpaca] {task_type}: {len(items)} samples collected")

    return categorized


# ---------------------------------------------------------------------------
# Step 3: Collect UIDs and Download Meshes
# ---------------------------------------------------------------------------

def collect_all_uids(
    pointllm_gt: List[Dict],
    alpaca_data: Dict[str, List[Dict]],
) -> Set[str]:
    """Collect all unique Objaverse UIDs needed for evaluation."""
    uids = set()

    for item in pointllm_gt:
        uid = item.get("object_id", "")
        if uid:
            uids.add(uid)

    for task_type, items in alpaca_data.items():
        for item in items:
            uid = item.get("uid", "")
            if uid and len(uid) > 5:
                uids.add(uid)

    print(f"[UIDs] Total unique UIDs to download: {len(uids)}")
    return uids


def download_objaverse_meshes(
    uids: Set[str],
    cache_dir: str,
    processes: int = 8,
) -> Dict[str, str]:
    """
    Download Objaverse meshes with retry logic and HF mirror fallback.

    Falls back to sequential download with retries if bulk download fails.
    """
    try:
        import objaverse
    except ImportError:
        print("[Error] Please install 'objaverse': pip install objaverse")
        sys.exit(1)

    uid_list = sorted(uids)
    print(f"[Objaverse] Downloading {len(uid_list)} meshes (processes={processes})...")

    # First try the standard objaverse library
    try:
        objects = objaverse.load_objects(uids=uid_list, download_processes=processes)
        uid_to_path = {
            uid: path for uid, path in objects.items()
            if path and os.path.exists(path)
        }
        print(f"[Objaverse] Successfully downloaded: {len(uid_to_path)} / {len(uid_list)}")
        return uid_to_path
    except Exception as e:
        print(f"[Objaverse] Bulk download failed: {e}")
        print(f"[Objaverse] Falling back to sequential download with retries...")

    # Fallback: get object paths first, then download individually via curl
    uid_to_path = {}
    try:
        object_paths = objaverse.load_object_paths()
    except Exception:
        print("[Objaverse] Cannot load object paths. Trying with fewer processes...")
        try:
            objects = objaverse.load_objects(uids=uid_list[:50], download_processes=1)
            uid_to_path = {
                uid: path for uid, path in objects.items()
                if path and os.path.exists(path)
            }
        except Exception:
            pass
        print(f"[Objaverse] Recovered {len(uid_to_path)} meshes in fallback mode")
        return uid_to_path

    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    base_dir = os.path.expanduser("~/.objaverse/hf-objaverse-v1")
    os.makedirs(base_dir, exist_ok=True)

    def download_one(uid: str) -> Optional[Tuple[str, str]]:
        if uid not in object_paths:
            return None
        obj_path = object_paths[uid]
        local_path = os.path.join(base_dir, obj_path)
        if os.path.exists(local_path):
            return (uid, local_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Try both HF and mirror
        urls = [
            f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{obj_path}",
            f"https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/{obj_path}",
        ]

        for url in urls:
            for attempt in range(3):
                try:
                    result = subprocess.run(
                        ["curl", "-sL", "--max-time", "30", url, "-o", local_path],
                        capture_output=True, timeout=45,
                    )
                    if result.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                        return (uid, local_path)
                except Exception:
                    pass
                import time
                time.sleep(0.5 * (attempt + 1))

        return None

    print(f"[Objaverse] Sequential download with curl fallback for {len(uid_list)} meshes...")
    with ThreadPoolExecutor(max_workers=min(processes, 4)) as executor:
        futures = {executor.submit(download_one, uid): uid for uid in uid_list}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                uid_to_path[result[0]] = result[1]
            if done % 100 == 0:
                print(f"[Objaverse] Progress: {done}/{len(uid_list)}, success: {len(uid_to_path)}")

    print(f"[Objaverse] Successfully downloaded: {len(uid_to_path)} / {len(uid_list)}")
    return uid_to_path


def scan_local_meshes(cache_dir: str) -> Dict[str, str]:
    """Scan local directory for already-downloaded meshes, building UID → path map."""
    uid_to_path = {}
    if not os.path.isdir(cache_dir):
        return uid_to_path

    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in MESH_EXTENSIONS:
                name = os.path.splitext(f)[0]
                full_path = os.path.join(root, f)
                uid_to_path[name] = full_path

    print(f"[LocalScan] Found {len(uid_to_path)} local mesh files")
    return uid_to_path


# ---------------------------------------------------------------------------
# Step 4: Build Understanding Dataset
# ---------------------------------------------------------------------------

def build_understanding_dataset(
    pointllm_gt: List[Dict],
    alpaca_caption_items: List[Dict],
    uid_to_path: Dict[str, str],
    output_path: str,
) -> int:
    """
    Build Understanding eval dataset from:
    1. PointLLM Objaverse Captioning (200 samples, official benchmark)
    2. 3D-Alpaca 3d_to_caption subset (additional samples for scale)
    """
    samples = []
    skipped = 0

    # --- Part 1: PointLLM benchmark ---
    for i, item in enumerate(pointllm_gt):
        uid = item["object_id"]
        if uid not in uid_to_path:
            skipped += 1
            continue

        gt_text = ""
        prompt = "Caption this 3D model in detail."
        for conv in item.get("conversations", []):
            if conv["from"] == "gpt":
                gt_text = conv["value"]
            if conv["from"] == "human":
                raw_prompt = conv["value"]
                prompt = raw_prompt.replace("<point>\n", "").replace("<point>", "").strip()
                if not prompt:
                    prompt = "Caption this 3D model in detail."

        samples.append({
            "sample_id": f"pointllm_{i:04d}",
            "mesh_path": uid_to_path[uid],
            "prompt": prompt,
            "ground_truth": gt_text,
            "ground_truths": [gt_text],
            "source": "pointllm_objaverse_captioning",
            "objaverse_uid": uid,
        })

    pointllm_count = len(samples)
    print(f"[Understanding] PointLLM: {pointllm_count} samples (skipped {skipped} missing meshes)")

    # --- Part 2: 3D-Alpaca caption subset ---
    existing_uids = {s["objaverse_uid"] for s in samples}
    alpaca_skipped = 0

    for i, item in enumerate(alpaca_caption_items):
        uid = item.get("uid", "")

        if uid in existing_uids:
            continue

        if uid not in uid_to_path:
            alpaca_skipped += 1
            continue

        human_msg = item["human_msg"]
        gpt_msg = item["gpt_msg"]

        prompt_clean = re.sub(r"<mesh-start>.*?<mesh-end>", "", human_msg)
        prompt_clean = re.sub(r"<mesh\d+>", "", prompt_clean).strip()
        if not prompt_clean:
            prompt_clean = "Caption this 3D model in detail."

        gt_text = re.sub(r"<mesh-start>.*?<mesh-end>", "", gpt_msg)
        gt_text = re.sub(r"<mesh\d+>", "", gt_text).strip()
        if not gt_text:
            alpaca_skipped += 1
            continue

        samples.append({
            "sample_id": f"alpaca_cap_{i:06d}",
            "mesh_path": uid_to_path[uid],
            "prompt": prompt_clean,
            "ground_truth": gt_text,
            "ground_truths": [gt_text],
            "source": "3d_alpaca_captioning",
            "objaverse_uid": uid,
        })
        existing_uids.add(uid)

    alpaca_count = len(samples) - pointllm_count
    print(f"[Understanding] 3D-Alpaca caption: {alpaca_count} samples (skipped {alpaca_skipped})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[Understanding] Total: {len(samples)} samples → {output_path}")
    return len(samples)


# ---------------------------------------------------------------------------
# Step 5: Build Generation Dataset
# ---------------------------------------------------------------------------

def build_generation_dataset(
    alpaca_gen_items: List[Dict],
    uid_to_path: Dict[str, str],
    output_path: str,
) -> int:
    """
    Build Generation eval dataset from 3D-Alpaca text-to-3D samples.

    If no 3D-Alpaca data is available, falls back to predefined prompts.
    """
    samples = []

    if alpaca_gen_items:
        for i, item in enumerate(alpaca_gen_items):
            human_msg = item["human_msg"]
            uid = item.get("uid", "")

            prompt = human_msg.strip()
            prompt = re.sub(
                r"^Please generate a 3D asset based on the prompt I provided:\s*",
                "",
                prompt,
            ).strip()
            if not prompt:
                prompt = human_msg.strip()

            sample: Dict[str, Any] = {
                "sample_id": f"alpaca_gen_{i:06d}",
                "prompt": prompt,
                "source": "3d_alpaca",
            }

            if uid and uid in uid_to_path:
                sample["reference_mesh_path"] = uid_to_path[uid]
                sample["objaverse_uid"] = uid

            samples.append(sample)
    else:
        print("[Generation] No 3D-Alpaca data available, using predefined prompts")
        for i, prompt in enumerate(GENERATION_PROMPTS):
            samples.append({
                "sample_id": f"builtin_gen_{i:04d}",
                "prompt": prompt,
                "source": "builtin_prompts",
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[Generation] Total: {len(samples)} samples → {output_path}")
    return len(samples)


# ---------------------------------------------------------------------------
# Step 6: Build VQVAE Reconstruction Dataset
# ---------------------------------------------------------------------------

def build_vqvae_dataset(
    uid_to_path: Dict[str, str],
    output_path: str,
    max_samples: Optional[int] = None,
) -> int:
    """
    Build VQVAE reconstruction dataset from all available meshes.

    Every downloaded mesh gets an encode-decode roundtrip evaluation.
    """
    samples = []

    for uid, path in sorted(uid_to_path.items()):
        ext = os.path.splitext(path)[1].lower()
        if ext not in MESH_EXTENSIONS:
            continue

        samples.append({
            "sample_id": f"vqvae_{uid[:16]}",
            "mesh_path": path,
            "objaverse_uid": uid,
        })

        if max_samples and len(samples) >= max_samples:
            break

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[VQVAE] Total: {len(samples)} samples → {output_path}")
    return len(samples)


# ---------------------------------------------------------------------------
# Step 7: Update YAML Configs
# ---------------------------------------------------------------------------

def update_yaml_configs(
    output_dir: str,
    eval_config_dir: str,
) -> None:
    """Update YAML task configs to point to generated data files."""
    import yaml

    updates = {
        "understanding.yaml": {
            "data": {
                "data_path": os.path.abspath(
                    os.path.join(output_dir, "understanding.json")
                ),
            },
            "reporting": {
                "output_dir": os.path.abspath(
                    os.path.join(output_dir, "../eval_results/understanding")
                ),
            },
        },
        "generation_text2mesh.yaml": {
            "data": {
                "data_path": os.path.abspath(
                    os.path.join(output_dir, "generation.json")
                ),
            },
            "reporting": {
                "output_dir": os.path.abspath(
                    os.path.join(output_dir, "../eval_results/generation")
                ),
            },
        },
        "vqvae_recon.yaml": {
            "data": {
                "data_path": os.path.abspath(
                    os.path.join(output_dir, "vqvae_recon.json")
                ),
            },
            "reporting": {
                "output_dir": os.path.abspath(
                    os.path.join(output_dir, "../eval_results/vqvae_recon")
                ),
            },
        },
    }

    tasks_dir = os.path.join(eval_config_dir, "tasks")
    for filename, overrides in updates.items():
        filepath = os.path.join(tasks_dir, filename)
        if not os.path.exists(filepath):
            print(f"[Config] Warning: {filepath} not found, skipping")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        for section, values in overrides.items():
            if section not in config:
                config[section] = {}
            config[section].update(values)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"[Config] Updated: {filepath}")


# ---------------------------------------------------------------------------
# Step 8: Print Summary
# ---------------------------------------------------------------------------

def print_summary(stats: Dict[str, int], output_dir: str) -> None:
    print("\n" + "=" * 70)
    print("  EVALUATION DATASET BUILD SUMMARY")
    print("=" * 70)
    for name, count in sorted(stats.items()):
        print(f"  {name:40s} : {count:6d} samples")
    print("-" * 70)
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".json"):
            fpath = os.path.join(output_dir, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"    - {f} ({size_mb:.1f} MB)")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Verify data:  python -c \"import json; d=json.load(open('eval_data/understanding.json')); print(len(d), d[0].keys())\"")
    print("  2. Run eval:     python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml")
    print("  3. Quick test:   python -m eval.runner --config eval/configs/tasks/understanding.yaml --max_samples 5")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build all evaluation datasets for ShapeLLM-Omni"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_data",
        help="Directory to save generated JSON datasets (default: eval_data)",
    )
    parser.add_argument(
        "--mesh_cache_dir",
        type=str,
        default=None,
        help="Directory for Objaverse mesh cache. If not set, uses ~/.objaverse default.",
    )
    parser.add_argument(
        "--max_alpaca_caption",
        type=int,
        default=1000,
        help="Max 3D-Alpaca captioning samples (default: 1000)",
    )
    parser.add_argument(
        "--max_alpaca_gen",
        type=int,
        default=1200,
        help="Max 3D-Alpaca text-to-3D samples (default: 1200)",
    )
    parser.add_argument(
        "--max_alpaca_editing",
        type=int,
        default=200,
        help="Max 3D-Alpaca 3D editing samples (default: 200)",
    )
    parser.add_argument(
        "--max_vqvae",
        type=int,
        default=None,
        help="Max VQVAE reconstruction samples (default: all available meshes)",
    )
    parser.add_argument(
        "--download_processes",
        type=int,
        default=8,
        help="Number of parallel download processes (default: 8)",
    )
    parser.add_argument(
        "--pointllm_scale",
        type=int,
        default=3000,
        choices=[200, 3000],
        help="PointLLM GT scale: 200 (small benchmark) or 3000 (large, default: 3000)",
    )
    parser.add_argument(
        "--extra_mesh_dirs",
        type=str,
        nargs="*",
        default=[],
        help="Additional directories to scan for local .glb/.obj mesh files",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip mesh download; only use already-cached meshes",
    )
    parser.add_argument(
        "--skip_alpaca",
        action="store_true",
        help="Skip 3D-Alpaca loading (use only PointLLM benchmark)",
    )
    parser.add_argument(
        "--update_configs",
        action="store_true",
        help="Update YAML config files with generated data paths",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: PointLLM GT ---
    print("\n" + "=" * 60)
    print("  Step 1/7: Download PointLLM Ground Truth")
    print("=" * 60)
    _, pointllm_gt = download_pointllm_gt(args.output_dir, scale=args.pointllm_scale)

    # --- Step 2: 3D-Alpaca ---
    alpaca_data: Dict[str, List[Dict]] = {
        "3d_to_caption": [],
        "text_to_3d": [],
        "3d_editing": [],
    }
    if not args.skip_alpaca:
        print("\n" + "=" * 60)
        print("  Step 2/7: Load 3D-Alpaca from HuggingFace")
        print("=" * 60)
        alpaca_data = load_3d_alpaca(
            cache_dir=args.output_dir,
            max_caption=args.max_alpaca_caption,
            max_gen=args.max_alpaca_gen,
            max_editing=args.max_alpaca_editing,
        )
    else:
        print("\n[Skip] 3D-Alpaca loading skipped")

    # --- Step 3: Collect UIDs ---
    print("\n" + "=" * 60)
    print("  Step 3/7: Collect Objaverse UIDs")
    print("=" * 60)
    all_uids = collect_all_uids(pointllm_gt, alpaca_data)

    # --- Step 4: Download / Scan Meshes ---
    print("\n" + "=" * 60)
    print("  Step 4/7: Acquire Objaverse Meshes")
    print("=" * 60)

    uid_to_path: Dict[str, str] = {}

    if args.mesh_cache_dir:
        uid_to_path.update(scan_local_meshes(args.mesh_cache_dir))

    for extra_dir in args.extra_mesh_dirs:
        if os.path.isdir(extra_dir):
            extra_meshes = scan_local_meshes(extra_dir)
            uid_to_path.update(extra_meshes)
        else:
            print(f"[Warning] Extra mesh dir not found: {extra_dir}")

    if not args.skip_download:
        missing_uids = all_uids - set(uid_to_path.keys())
        if missing_uids:
            print(f"[Download] {len(missing_uids)} meshes not found locally, downloading...")
            try:
                downloaded = download_objaverse_meshes(
                    missing_uids,
                    args.mesh_cache_dir or os.path.expanduser("~/.objaverse"),
                    processes=args.download_processes,
                )
                uid_to_path.update(downloaded)
            except Exception as e:
                print(f"[Download] Download encountered error: {e}")
                print(f"[Download] Continuing with {len(uid_to_path)} available meshes...")
        else:
            print(f"[Download] All {len(all_uids)} meshes already cached locally")
    else:
        print(f"[Skip] Download skipped. Using {len(uid_to_path)} cached meshes.")

    # --- Step 5: Build Understanding Dataset ---
    print("\n" + "=" * 60)
    print("  Step 5/7: Build Understanding Dataset")
    print("=" * 60)
    understanding_count = build_understanding_dataset(
        pointllm_gt,
        alpaca_data.get("3d_to_caption", []),
        uid_to_path,
        os.path.join(args.output_dir, "understanding.json"),
    )

    # --- Step 6: Build Generation Dataset ---
    print("\n" + "=" * 60)
    print("  Step 6/7: Build Generation Dataset")
    print("=" * 60)
    generation_count = build_generation_dataset(
        alpaca_data.get("text_to_3d", []),
        uid_to_path,
        os.path.join(args.output_dir, "generation.json"),
    )

    # --- Step 7: Build VQVAE Dataset ---
    print("\n" + "=" * 60)
    print("  Step 7/7: Build VQVAE Reconstruction Dataset")
    print("=" * 60)
    vqvae_count = build_vqvae_dataset(
        uid_to_path,
        os.path.join(args.output_dir, "vqvae_recon.json"),
        max_samples=args.max_vqvae,
    )

    # --- Update configs ---
    if args.update_configs:
        print("\n[Config] Updating YAML configs...")
        project_root = Path(__file__).resolve().parent.parent.parent
        eval_config_dir = project_root / "eval" / "configs"
        update_yaml_configs(args.output_dir, str(eval_config_dir))

    # --- Summary ---
    stats = {
        "Understanding (captioning/QA)": understanding_count,
        "Generation (text-to-3D)": generation_count,
        "VQVAE Reconstruction": vqvae_count,
        "Total unique meshes downloaded": len(uid_to_path),
    }
    print_summary(stats, args.output_dir)

    # Save metadata
    meta = {
        "total_meshes": len(uid_to_path),
        "understanding_samples": understanding_count,
        "generation_samples": generation_count,
        "vqvae_samples": vqvae_count,
        "pointllm_gt_count": len(pointllm_gt),
        "alpaca_caption_count": len(alpaca_data.get("3d_to_caption", [])),
        "alpaca_gen_count": len(alpaca_data.get("text_to_3d", [])),
    }
    meta_path = os.path.join(args.output_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Meta] Dataset metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
