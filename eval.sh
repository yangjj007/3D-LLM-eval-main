export HF_HOME=/yangjunjie/3D-LLM-eval-main/eval_data/hf_cache
export HUGGINGFACE_HUB_CACHE=/yangjunjie/3D-LLM-eval-main/eval_data/hf_cache
export SENTENCE_TRANSFORMERS_HOME=/yangjunjie/3D-LLM-eval-main/eval_data/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml --adapter shapellm --gpu_ids 0
python -m eval.runner --config eval/configs/tasks/understanding.yaml --adapter shapellm --gpu_ids 0
python -m eval.runner --config eval/configs/tasks/generation_text2mesh.yaml --adapter shapellm --gpu_ids 0

python -m eval.runner --config eval/configs/tasks/sparse_vqvae_recon.yaml --adapter sparse_sdf_qwen3 --gpu_ids 0 --batch_size 1
python -m eval.runner --config eval/configs/tasks/sparse_understanding.yaml --adapter sparse_sdf_qwen3 --gpu_ids 0 --batch_size 1
python -m eval.runner --config eval/configs/tasks/sparse_generation.yaml --adapter sparse_sdf_qwen3 --gpu_ids 0 --batch_size 1