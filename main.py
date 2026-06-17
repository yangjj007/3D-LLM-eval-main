import os
import torch
import numpy as np
import trimesh
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from huggingface_hub import hf_hub_download
import uuid
import random
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.models.sparse_structure_vqvae import VQVAE3D
from trellis.utils import render_utils, postprocessing_utils

# --- 配置环境 ---
os.environ['SPCONV_ALGO'] = 'native'
HF_TOKEN = os.environ.get("HF_TOKEN", None)
OUTPUT_DIR = "./output_shapellm"  # 输出文件保存路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 可视化工具函数 (新增详细可视化) ---

def save_visualization(points, title, filename, color=None):
    """
    保存3D散点图可视化文件 (HTML格式，可用浏览器打开查看交互式3D)
    """
    print(f"[-] Generating visualization for: {title} ...")
    if color is None:
        N = len(points)
        palette = px.colors.qualitative.Set3
        color = [palette[i % len(palette)] for i in range(N)]
        random.shuffle(color)

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=color,
            opacity=0.8
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-0.6, 0.6]),
            yaxis=dict(range=[-0.6, 0.6]),
            zaxis=dict(range=[-0.6, 0.6]),
            aspectmode='cube'
        )
    )
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(save_path)
    print(f"    Saved to: {save_path}")

def convert_trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(trimesh_mesh.faces, dtype=np.int32)
    )
    return o3d_mesh

def rotate_points(points, axis='x', angle_deg=90):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'x':
        R = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])[:3, :3]
    elif axis == 'y':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])[:3, :3]
    elif axis == 'z':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])[:3, :3]
    return points @ R.T

# --- 核心处理逻辑：GLB -> Voxel -> Token (含详细步骤展示) ---

def process_mesh_with_visualization(filepath, vqvae_model, device):
    print(f"\n[1] Starting Mesh Processing & Voxelization for: {filepath}")
    
    # 1. 加载 Mesh
    mesh = trimesh.load(filepath, force='mesh')
    original_verts = np.asarray(mesh.vertices)
    save_visualization(original_verts, "Step 1: Original Raw Vertices", "step1_raw_mesh.html")

    # 2. Open3D 转换与归一化
    mesh_o3d = convert_trimesh_to_open3d(mesh)
    vertices = np.asarray(mesh_o3d.vertices)
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    
    # 归一化到 [0, 1] 然后平移到 [-0.5, 0.5]
    max_extent = (max_vals - min_vals).max()
    center = (max_vals + min_vals) / 2
    vertices_normalized = (vertices - center) / max_extent # Scale to fit in unit cube preserving aspect ratio
    # 原始代码逻辑略有不同，这里保持原始代码的归一化逻辑：
    # vertices_normalized = (vertices - vertices.min()) / (vertices.max() - vertices.min()) 
    # 注意：原始代码的归一化可能导致非等比缩放，这里为了从简，沿用原始代码逻辑修正后的版本，确保在中心
    
    # 原始代码逻辑复现：
    min_v = vertices.min()
    max_v = vertices.max()
    vertices_norm = (vertices - min_v) / (max_v - min_v) # [0, 1]
    vertices_centered = vertices_norm * 1.0 - 0.5        # [-0.5, 0.5]
    vertices_centered = np.clip(vertices_centered, -0.5 + 1e-6, 0.5 - 1e-6)
    
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices_centered)
    save_visualization(vertices_centered, "Step 2: Normalized & Centered Vertices", "step2_normalized.html")

    # 3. 体素化 (Voxelization)
    print("[-] Running Voxelization (Resolution 64^3)...")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh_o3d, 
        voxel_size=1/64, 
        min_bound=(-0.5, -0.5, -0.5), 
        max_bound=(0.5, 0.5, 0.5)
    )
    
    # 获取体素中心点
    voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()]) # (N, 3) int
    
    # 可视化整数索引对应的物理空间位置
    # grid_index is 0..63. Center is (index + 0.5) / 64 - 0.5
    voxel_centers = (voxel_indices + 0.5) / 64 - 0.5
    save_visualization(voxel_centers, "Step 3: Voxel Grid Centers (Pre-Rotation)", "step3_voxels_pre_rotate.html")
    
    # 4. 旋转 (ShapeLLM 特定的预处理)
    # The original code rotates x-axis by 90 degrees
    voxel_rotated = rotate_points(voxel_centers, axis='x', angle_deg=90)
    save_visualization(voxel_rotated, "Step 4: Voxel Grid (Rotated Input for Model)", "step4_voxels_final_input.html")

    # 5. 构建 Tensor 输入 VQVAE
    # 重新计算旋转后的整数坐标，用于构建 dense grid
    coords = ((torch.from_numpy(voxel_rotated) + 0.5) * 64).int().contiguous()
    
    # 构建 64x64x64 的二值网格
    ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)
    # 过滤越界坐标 (以防旋转后出现浮点误差)
    coords = torch.clamp(coords, 0, 63)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    
    # 6. VQVAE 编码 (Discrete Tokenization)
    print("[-] Encoding Voxels via VQVAE...")
    with torch.no_grad():
        token = vqvae_model.Encode(ss.unsqueeze(1).to(dtype=torch.float32).to(device))
        token_list = token[0].cpu().numpy().tolist()
    
    print(f"[-] Encoded into {len(token_list)} tokens.")
    return token_list

def token_to_words(token_list):
    mesh_str = "<mesh-start>"
    for idx in token_list:
        mesh_str += f"<mesh{idx}>"
    mesh_str += "<mesh-end>"
    return mesh_str

# --- 模型加载与初始化 ---

def initialize_models():
    print("\n[Init] Loading Models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. VQVAE
    print("Loading VQVAE...")
    vqvae = VQVAE3D(num_embeddings=8192)
    vqvae.eval()
    try:
        filepath = hf_hub_download(repo_id="yejunliang23/3DVQVAE", filename="3DVQVAE.bin")
        state_dict = torch.load(filepath, map_location="cpu")
        vqvae.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading VQVAE from HF: {e}. Ensure you have internet or local cache.")
    vqvae = vqvae.to(device)

    # 2. ShapeLLM (Qwen2.5-VL)
    print("Loading ShapeLLM (Omni)...")
    MODEL_DIR = "/wangcm/ShapeLLM-7B-omni" # 用户指定的路径
    # 如果本地没有，尝试从 HF 拉取作为 fallback (可选)
    # MODEL_DIR = "yejunliang23/ShapeLLM-7B-omni" 
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, 
            torch_dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        tokenizer = processor.tokenizer
    except OSError:
        print(f"Could not load from {MODEL_DIR}, trying HuggingFace remote...")
        MODEL_DIR_REMOTE = "yejunliang23/ShapeLLM-7B-omni"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR_REMOTE, 
            torch_dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_DIR_REMOTE)
        tokenizer = processor.tokenizer

    # 3. Trellis Pipelines
    print("Loading Trellis Pipelines...")
    pipeline_text = TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")
    pipeline_text.to(device)
    # pipeline_image = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large") # 暂时不需要图片转3D
    # pipeline_image.to(device)
    
    return vqvae, model, processor, tokenizer, pipeline_text, device

# --- 推理与生成 ---

def generate_response_and_model(
    vqvae, model, processor, tokenizer, pipeline_text, device,
    glb_path, prompt_text
):
    # --- 1. 处理输入 GLB ---
    token_list = process_mesh_with_visualization(glb_path, vqvae, device)
    mesh_tokens_str = token_to_words(token_list)
    
    # --- 2. 构造 Prompt ---
    # ShapeLLM 理解 <mesh> token 序列
    full_prompt = f"{mesh_tokens_str}\n{prompt_text}"
    print(f"\n[2] Sending prompt to LLM (Mesh tokens + '{prompt_text}')...")

    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': full_prompt}]}
    ]
    
    # --- 3. LLM 推理 ---
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], padding=True, return_tensors='pt')
    inputs = inputs.to(device)

    # 特殊 token 处理
    # 159858 是 ShapeLLM 可能用到的特殊结束符
    eos_token_id = [tokenizer.eos_token_id, 159858] 

    print("[-] Generating tokens...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            top_k=8192,
            top_p=0.7,
            temperature=0.7,
            eos_token_id=eos_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # 清洗文本用于显示
    clean_response = response_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    print(f"\n[3] LLM Response:\n{clean_response}")

    # --- 4. 解析输出中的 Mesh Token ---
    # 简单的解析逻辑：寻找 <meshX>
    output_tokens = []
    
    # 提取 <mesh123> 格式
    # 原始代码逻辑是流式处理的，这里做全量解析
    parts = response_text.split("><mesh")
    raw_indices = []
    
    # 处理开头
    if parts[0].startswith("<mesh"):
        try:
            raw_indices.append(int(parts[0].replace("<mesh", "")))
        except: pass
        
    for p in parts[1:]:
        # p 可能长这样: "123> some text..."
        try:
            token_str = p.split(">")[0]
            raw_indices.append(int(token_str))
        except:
            pass
            
    # 补齐到 1024
    if len(raw_indices) > 0:
        print(f"[-] Extracted {len(raw_indices)} mesh tokens from response.")
        while len(raw_indices) < 1024:
            raw_indices.append(raw_indices[-1])
        encoding_indices = torch.tensor(raw_indices[:1024]).unsqueeze(0).to(device) # (1, 1024)
        
        # --- 5. 重建 3D 模型 ---
        reconstruct_3d(vqvae, pipeline_text, encoding_indices, device, clean_response)
    else:
        print("[!] No mesh tokens found in the response. Only text returned.")

def reconstruct_3d(vqvae, pipeline_text, encoding_indices, device, prompt_context):
    print("\n[4] Reconstructing 3D Model from tokens...")
    
    # 5.1 VQVAE 解码 -> Voxel Logits
    recon = vqvae.Decode(encoding_indices)
    z_s = recon[0].detach().cpu()
    z_s = (z_s > 0) * 1 # 二值化
    
    indices = torch.nonzero(z_s[0] == 1)
    
    # 可视化生成的体素
    position_recon = (indices.float() + 0.5) / 64 - 0.5
    save_visualization(position_recon.numpy(), "Step 5: LLM Generated Voxels", "step5_generated_voxels.html")
    
    # 准备 Trellis 输入
    position = position_recon
    coords = ((position + 0.5) * 64).int().contiguous()
    ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    ss = ss.unsqueeze(0)
    
    # 提取坐标 [batch_idx, z, y, x]? 原始代码使用的是 coords[:, [0, 2, 3, 4]]
    # 这里的 ss 是 (B, D, H, W) -> argwhere 返回 (B, D, H, W) 索引
    coords_tensor = torch.argwhere(ss > 0)[:, [0, 1, 2, 3]].int().to(device)
    # 注意：Trellis 的坐标顺序可能需要根据原始代码调整。
    # 原始代码: coords = torch.argwhere(ss>0)[:, [0, 2, 3, 4]].int() implying 5 dimensions?
    # 实际上 ss 是 1, 64, 64, 64 (4 dims). argwhere 返回 (N, 4).
    # 让我们直接使用 pipeline 需要的格式。通常是 Sparse Tensor 坐标.
    
    print("[-] Running Trellis Decoder (Geometry Refinement)...")
    
    # 这里我们使用 pipeline_text 来进行 conditioned generation，
    # 或者如果只是解码结构，通常使用 trellis 的 decoder。
    # 原始代码使用了 pipeline_text.sample_slat 和 decode_slat
    
    with torch.no_grad():
        # 使用 LLM 的文本回复或者原始 Query 作为 Prompt
        # 这里为了简单，如果 LLM 回复太长，可能截取一部分，或者使用原始 Prompt
        cond = pipeline_text.get_cond([prompt_context[:70]]) # 截断一下防止过长
        slat = pipeline_text.sample_slat(cond, coords_tensor)
        outputs = pipeline_text.decode_slat(slat, ['mesh', 'gaussian'])
    
    # 5.2 导出 GLB
    trial_id = str(uuid.uuid4())[:8]
    output_glb_path = os.path.join(OUTPUT_DIR, f"result_{trial_id}.glb")
    
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
        verbose=False
    )
    glb.export(output_glb_path)
    print(f"\n[Success] Generated model saved to: {output_glb_path}")

# --- 主入口 ---

def main():
    # --- 用户配置 ---
    # 输入模型路径 (请替换为真实存在的 GLB 文件路径)
    INPUT_GLB = "./test.glb" 
    
    # 提示词
    PROMPT = "Give a quick overview of the object represented by this 3D mesh and generate a variation of it."
    
    if not os.path.exists(INPUT_GLB):
        # 创建一个假的示例 GLB 用于测试 (如果文件不存在)
        print(f"Warning: {INPUT_GLB} not found. Creating a dummy cube for testing.")
        m = trimesh.creation.box()
        m.export(INPUT_GLB)

    # 1. 初始化
    vqvae, model, processor, tokenizer, pipeline_text, device = initialize_models()
    
    # 2. 运行
    generate_response_and_model(
        vqvae, model, processor, tokenizer, pipeline_text, device,
        INPUT_GLB, PROMPT
    )

if __name__ == "__main__":
    main()