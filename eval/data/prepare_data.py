"""
Data preparation scripts for ShapeLLM-Omni evaluation.

Downloads and converts benchmark datasets into the format required by the eval framework.
Supports:
  - PointLLM Objaverse Captioning benchmark (3D-to-Caption)
  - Toys4K generation benchmark (Text-to-3D / Image-to-3D)
  - 3D-Alpaca (from HuggingFace)
  - Custom dataset generation from local mesh files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. PointLLM Objaverse Captioning Benchmark
# ---------------------------------------------------------------------------

POINTLLM_REPO = "https://github.com/InternRobotics/PointLLM"
POINTLLM_GT_URL = (
    "https://raw.githubusercontent.com/InternRobotics/PointLLM/main/"
    "data/PointLLM_brief_description_val_200_GT.json"
)


def prepare_pointllm_captioning(
    output_path: str,
    objaverse_mesh_dir: str,
    gt_json_path: Optional[str] = None,
    mesh_format: str = "glb",
) -> str:
    """
    Prepare the PointLLM Objaverse captioning benchmark for ShapeLLM-Omni eval.

    The PointLLM benchmark uses a curated 200-sample test set from Objaverse.
    ShapeLLM-Omni uses the same test set but with mesh input instead of point clouds.

    Args:
        output_path: Where to save the converted JSON file.
        objaverse_mesh_dir: Directory containing Objaverse mesh files,
            organized as {objaverse_mesh_dir}/{uid}.{mesh_format} or
            {objaverse_mesh_dir}/{uid}/{uid}.{mesh_format}
        gt_json_path: Path to PointLLM_brief_description_val_200_GT.json.
            If None, downloads from GitHub.
        mesh_format: Mesh file extension (glb, obj, ply).

    Returns:
        Path to the generated JSON file.

    PointLLM GT format:
    [
      {
        "object_id": "xxxx...",  // Objaverse UID
        "conversations": [
          {"from": "human", "value": "<point>\nCaption this 3D model in detail."},
          {"from": "gpt", "value": "This is a ..."}
        ]
      }
    ]

    Output format (for eval framework):
    [
      {
        "sample_id": "pointllm_000",
        "mesh_path": "/path/to/uid.glb",
        "prompt": "Caption this 3D model in detail.",
        "ground_truth": "This is a ...",
        "ground_truths": ["This is a ..."],
        "source": "pointllm_objaverse_captioning",
        "objaverse_uid": "xxxx..."
      }
    ]
    """
    if gt_json_path is None:
        gt_json_path = _download_pointllm_gt(output_path)

    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    samples = []
    skipped = 0
    for i, item in enumerate(gt_data):
        uid = item["object_id"]
        mesh_path = _find_mesh(objaverse_mesh_dir, uid, mesh_format)

        if mesh_path is None:
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
            "mesh_path": str(mesh_path),
            "prompt": prompt,
            "ground_truth": gt_text,
            "ground_truths": [gt_text],
            "source": "pointllm_objaverse_captioning",
            "objaverse_uid": uid,
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[DataPrep] PointLLM Captioning: {len(samples)} samples saved, {skipped} skipped (mesh not found)")
    print(f"[DataPrep] Output: {output_path}")
    return output_path


def _download_pointllm_gt(output_dir: str) -> str:
    """Download PointLLM ground truth JSON."""
    import urllib.request

    gt_path = os.path.join(
        os.path.dirname(output_dir) or ".",
        "PointLLM_brief_description_val_200_GT.json",
    )
    if not os.path.exists(gt_path):
        print(f"[DataPrep] Downloading PointLLM GT from GitHub...")
        urllib.request.urlretrieve(POINTLLM_GT_URL, gt_path)
        print(f"[DataPrep] Saved to: {gt_path}")
    return gt_path


# ---------------------------------------------------------------------------
# 2. Toys4K Generation Benchmark
# ---------------------------------------------------------------------------

def prepare_toys4k_generation(
    output_path: str,
    toys4k_dir: str,
    split: str = "test",
    task: str = "text_to_3d",
) -> str:
    """
    Prepare the Toys4K benchmark for generation evaluation.

    ShapeLLM-Omni paper uses Toys4K test set for text-to-3D and image-to-3D.
    Metrics: FD (Frechet Distance), KD (Kernel Distance), CLIP Score.

    Args:
        output_path: Where to save the converted JSON file.
        toys4k_dir: Root directory of Toys4K dataset.
        split: Dataset split (test).
        task: 'text_to_3d' or 'image_to_3d'.

    Toys4K structure (expected):
        toys4k_dir/
        ├── test/
        │   ├── metadata.json or captions.json
        │   ├── meshes/
        │   │   ├── 0001.glb
        │   │   └── ...
        │   └── images/  (for image-to-3d)
        │       ├── 0001.png
        │       └── ...

    Output format (text-to-3d):
    [
      {
        "sample_id": "toys4k_0001",
        "prompt": "A small yellow toy duck",
        "reference_mesh_path": "/path/to/0001.glb"
      }
    ]
    """
    split_dir = os.path.join(toys4k_dir, split)

    # Try to find metadata/captions
    meta_candidates = [
        os.path.join(split_dir, "metadata.json"),
        os.path.join(split_dir, "captions.json"),
        os.path.join(toys4k_dir, "metadata.json"),
        os.path.join(toys4k_dir, "captions.json"),
    ]
    meta_path = None
    for p in meta_candidates:
        if os.path.exists(p):
            meta_path = p
            break

    samples = []

    if meta_path:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if isinstance(metadata, dict):
            items = list(metadata.items())
        elif isinstance(metadata, list):
            items = [(str(i), v) for i, v in enumerate(metadata)]

        for key, val in items:
            if isinstance(val, str):
                caption = val
                mesh_name = key
            elif isinstance(val, dict):
                caption = val.get("caption", val.get("text", val.get("description", "")))
                mesh_name = val.get("mesh", val.get("id", key))
            else:
                continue

            mesh_path = _find_mesh_in_dirs(
                [os.path.join(split_dir, "meshes"), split_dir, toys4k_dir],
                mesh_name,
            )

            sample = {
                "sample_id": f"toys4k_{key}",
                "prompt": caption,
            }

            if mesh_path:
                sample["reference_mesh_path"] = str(mesh_path)

            if task == "image_to_3d":
                img_path = _find_image(
                    [os.path.join(split_dir, "images"), split_dir],
                    mesh_name,
                )
                if img_path:
                    sample["image_path"] = str(img_path)

            samples.append(sample)
    else:
        # No metadata: scan for meshes and use filenames as prompts
        mesh_dir = os.path.join(split_dir, "meshes")
        if not os.path.isdir(mesh_dir):
            mesh_dir = split_dir

        for f in sorted(os.listdir(mesh_dir)):
            if f.endswith((".glb", ".obj", ".ply", ".gltf")):
                name = os.path.splitext(f)[0]
                samples.append({
                    "sample_id": f"toys4k_{name}",
                    "prompt": name.replace("_", " ").replace("-", " "),
                    "reference_mesh_path": os.path.join(mesh_dir, f),
                })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[DataPrep] Toys4K ({task}): {len(samples)} samples saved")
    print(f"[DataPrep] Output: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 3. 3D-Alpaca Dataset (HuggingFace)
# ---------------------------------------------------------------------------

def prepare_3d_alpaca(
    output_path: str,
    hf_dataset_path: str = "yejunliang23/3D-Alpaca",
    task_filter: Optional[str] = None,
    max_samples: Optional[int] = None,
    objaverse_mesh_dir: Optional[str] = None,
) -> str:
    """
    Prepare evaluation data from the 3D-Alpaca HuggingFace dataset.

    Downloads and converts 3D-Alpaca into eval framework format.

    Args:
        output_path: Where to save the converted JSON file.
        hf_dataset_path: HuggingFace dataset repository.
        task_filter: Filter by task type ('text_to_3d', 'image_to_3d',
            '3d_to_caption', '3d_editing'). None = all tasks.
        max_samples: Maximum samples to include.
        objaverse_mesh_dir: Directory with Objaverse meshes (to resolve UIDs to paths).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] Install 'datasets' library: pip install datasets")
        return ""

    print(f"[DataPrep] Loading 3D-Alpaca from {hf_dataset_path}...")
    ds = load_dataset(hf_dataset_path)

    # 3D-Alpaca likely has train split; use it for eval subset
    if "test" in ds:
        data = ds["test"]
    elif "validation" in ds:
        data = ds["validation"]
    else:
        data = ds["train"]

    samples = []
    for i, item in enumerate(data):
        if max_samples and len(samples) >= max_samples:
            break

        # Detect task type from conversations
        conversations = item.get("conversations", [])
        if not conversations:
            continue

        human_msg = ""
        gpt_msg = ""
        for conv in conversations:
            if conv.get("from") == "human":
                human_msg = conv.get("value", "")
            elif conv.get("from") == "gpt":
                gpt_msg = conv.get("value", "")

        has_mesh_input = "<mesh-start>" in human_msg or "<mesh" in human_msg
        has_mesh_output = "<mesh-start>" in gpt_msg or "<mesh" in gpt_msg

        if has_mesh_input and not has_mesh_output:
            task_type = "3d_to_caption"
        elif not has_mesh_input and has_mesh_output:
            task_type = "text_to_3d"
        elif has_mesh_input and has_mesh_output:
            task_type = "3d_editing"
        else:
            task_type = "text_only"

        if task_filter and task_type != task_filter:
            continue

        uid = item.get("object_id", item.get("uid", f"alpaca_{i}"))

        sample: Dict[str, Any] = {
            "sample_id": f"alpaca_{i:06d}",
            "task_type": task_type,
            "objaverse_uid": uid,
        }

        if task_type == "3d_to_caption":
            # Extract prompt (strip mesh tokens from human message)
            import re
            prompt_clean = re.sub(r"<mesh-start>.*?<mesh-end>", "", human_msg).strip()
            prompt_clean = re.sub(r"<mesh\d+>", "", prompt_clean).strip()
            if not prompt_clean:
                prompt_clean = "Caption this 3D model in detail."

            sample["prompt"] = prompt_clean
            sample["ground_truth"] = gpt_msg
            sample["ground_truths"] = [gpt_msg]

            if objaverse_mesh_dir:
                mesh_path = _find_mesh(objaverse_mesh_dir, uid, "glb")
                if mesh_path:
                    sample["mesh_path"] = str(mesh_path)

        elif task_type == "text_to_3d":
            sample["prompt"] = human_msg

        elif task_type == "3d_editing":
            import re
            prompt_clean = re.sub(r"<mesh-start>.*?<mesh-end>", "", human_msg).strip()
            prompt_clean = re.sub(r"<mesh\d+>", "", prompt_clean).strip()
            sample["prompt"] = prompt_clean

            if objaverse_mesh_dir:
                mesh_path = _find_mesh(objaverse_mesh_dir, uid, "glb")
                if mesh_path:
                    sample["mesh_path"] = str(mesh_path)

        samples.append(sample)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[DataPrep] 3D-Alpaca ({task_filter or 'all'}): {len(samples)} samples saved")
    print(f"[DataPrep] Output: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 4. Custom Dataset from Local Mesh Directory
# ---------------------------------------------------------------------------

def prepare_from_mesh_dir(
    output_path: str,
    mesh_dir: str,
    task: str = "vqvae_recon",
    captions_json: Optional[str] = None,
    default_prompt: str = "Caption this 3D model in detail.",
) -> str:
    """
    Generate evaluation data from a local directory of mesh files.

    Useful for VQVAE reconstruction evaluation or quick captioning tests.

    Args:
        output_path: Where to save the JSON file.
        mesh_dir: Directory containing mesh files (.glb, .obj, .ply, .stl).
        task: 'vqvae_recon', 'understanding', or 'generation'.
        captions_json: Optional JSON mapping filename → caption
            (for understanding task).
        default_prompt: Default prompt for understanding task.
    """
    captions = {}
    if captions_json and os.path.exists(captions_json):
        with open(captions_json, "r", encoding="utf-8") as f:
            captions = json.load(f)

    mesh_extensions = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}
    samples = []

    for f in sorted(os.listdir(mesh_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in mesh_extensions:
            continue

        name = os.path.splitext(f)[0]
        mesh_path = os.path.join(mesh_dir, f)

        if task == "vqvae_recon":
            samples.append({
                "sample_id": f"local_{name}",
                "mesh_path": mesh_path,
            })
        elif task == "understanding":
            caption = captions.get(name, captions.get(f, ""))
            samples.append({
                "sample_id": f"local_{name}",
                "mesh_path": mesh_path,
                "prompt": default_prompt,
                "ground_truth": caption,
                "ground_truths": [caption] if caption else [],
            })
        elif task == "generation":
            caption = captions.get(name, captions.get(f, name.replace("_", " ")))
            samples.append({
                "sample_id": f"local_{name}",
                "prompt": caption,
                "reference_mesh_path": mesh_path,
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[DataPrep] Local meshes ({task}): {len(samples)} samples saved")
    print(f"[DataPrep] Output: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 5. Download Objaverse Meshes
# ---------------------------------------------------------------------------

def download_objaverse_meshes(
    uids: List[str],
    output_dir: str,
    processes: int = 4,
) -> Dict[str, str]:
    """
    Download Objaverse meshes by UID using the objaverse Python package.

    Args:
        uids: List of Objaverse UIDs to download.
        output_dir: Directory to save downloaded meshes.
        processes: Number of parallel download processes.

    Returns:
        Dict mapping UID → local file path.
    """
    try:
        import objaverse
    except ImportError:
        print("[Error] Install objaverse: pip install objaverse")
        return {}

    print(f"[DataPrep] Downloading {len(uids)} meshes from Objaverse...")
    os.makedirs(output_dir, exist_ok=True)

    objects = objaverse.load_objects(uids=uids, download_processes=processes)
    uid_to_path = {}
    for uid, path in objects.items():
        uid_to_path[uid] = path

    print(f"[DataPrep] Downloaded {len(uid_to_path)} meshes to {output_dir}")
    return uid_to_path


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _find_mesh(
    base_dir: str, uid: str, fmt: str = "glb"
) -> Optional[Path]:
    """Find a mesh file by Objaverse UID in various directory structures."""
    candidates = [
        Path(base_dir) / f"{uid}.{fmt}",
        Path(base_dir) / uid / f"{uid}.{fmt}",
        Path(base_dir) / uid[:2] / f"{uid}.{fmt}",
        Path(base_dir) / f"{uid}.obj",
        Path(base_dir) / f"{uid}.ply",
        Path(base_dir) / uid / f"model.{fmt}",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_mesh_in_dirs(
    dirs: List[str], name: str
) -> Optional[Path]:
    """Search for a mesh in multiple directories."""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for ext in [".glb", ".obj", ".ply", ".gltf", ".stl"]:
            p = Path(d) / f"{name}{ext}"
            if p.exists():
                return p
            p = Path(d) / name / f"{name}{ext}"
            if p.exists():
                return p
    return None


def _find_image(
    dirs: List[str], name: str
) -> Optional[Path]:
    """Search for an image in multiple directories."""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            p = Path(d) / f"{name}{ext}"
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation datasets for ShapeLLM-Omni"
    )
    subparsers = parser.add_subparsers(dest="command", help="Dataset type")

    # PointLLM captioning
    p1 = subparsers.add_parser("pointllm", help="PointLLM Objaverse captioning benchmark")
    p1.add_argument("--output", type=str, required=True, help="Output JSON path")
    p1.add_argument("--mesh_dir", type=str, required=True, help="Objaverse mesh directory")
    p1.add_argument("--gt_json", type=str, default=None, help="Path to PointLLM GT JSON")
    p1.add_argument("--mesh_format", type=str, default="glb", help="Mesh file extension")

    # Toys4K generation
    p2 = subparsers.add_parser("toys4k", help="Toys4K generation benchmark")
    p2.add_argument("--output", type=str, required=True, help="Output JSON path")
    p2.add_argument("--toys4k_dir", type=str, required=True, help="Toys4K root directory")
    p2.add_argument("--split", type=str, default="test")
    p2.add_argument("--task", type=str, default="text_to_3d", choices=["text_to_3d", "image_to_3d"])

    # 3D-Alpaca
    p3 = subparsers.add_parser("alpaca", help="3D-Alpaca from HuggingFace")
    p3.add_argument("--output", type=str, required=True, help="Output JSON path")
    p3.add_argument("--hf_path", type=str, default="yejunliang23/3D-Alpaca")
    p3.add_argument("--task_filter", type=str, default=None,
                     choices=["text_to_3d", "image_to_3d", "3d_to_caption", "3d_editing"])
    p3.add_argument("--max_samples", type=int, default=None)
    p3.add_argument("--mesh_dir", type=str, default=None, help="Objaverse mesh directory")

    # Local mesh directory
    p4 = subparsers.add_parser("local", help="Local mesh directory")
    p4.add_argument("--output", type=str, required=True, help="Output JSON path")
    p4.add_argument("--mesh_dir", type=str, required=True, help="Mesh directory")
    p4.add_argument("--task", type=str, default="vqvae_recon",
                     choices=["vqvae_recon", "understanding", "generation"])
    p4.add_argument("--captions_json", type=str, default=None, help="Optional captions mapping")
    p4.add_argument("--prompt", type=str, default="Caption this 3D model in detail.")

    args = parser.parse_args()

    if args.command == "pointllm":
        prepare_pointllm_captioning(args.output, args.mesh_dir, args.gt_json, args.mesh_format)
    elif args.command == "toys4k":
        prepare_toys4k_generation(args.output, args.toys4k_dir, args.split, args.task)
    elif args.command == "alpaca":
        prepare_3d_alpaca(args.output, args.hf_path, args.task_filter, args.max_samples, args.mesh_dir)
    elif args.command == "local":
        prepare_from_mesh_dir(args.output, args.mesh_dir, args.task, args.captions_json, args.prompt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
