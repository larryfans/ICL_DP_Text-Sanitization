export CUDA_VISIBLE_DEVICES=2

# python llama_dolly_noise.py --eps 1 \
#     --top_k 20 \
#     --RoPE True \
#     --combine_method decode

# python llama_dolly_noise.py --eps 2 \
#     --top_k 20 \
#     --RoPE True \
#     --combine_method decode

# python llama_dolly_noise.py --eps 3 \
#     --top_k 20 \
#     --RoPE True \
#     --combine_method decode


python llm_infer.py --version_folder 'Noise_version' \
    --version 'cm_decode_RoPE_True_eps_1.0_top_20' \
    --shot 2 \
    --save_folder './Eval_Final/'

python llm_infer.py --version_folder '.' \
    --version 'cm_decode_RoPE_True_eps_2.0_top_20' \
    --shot 2 \
    --save_folder './Eval_Final/'

python llm_infer.py --version_folder '.' \
    --version 'cm_decode_RoPE_True_eps_3.0_top_20' \
    --shot 2 \
    --save_folder './Eval_Final/'

