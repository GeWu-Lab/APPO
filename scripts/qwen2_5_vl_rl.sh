qwen2_5_vl_ckpt_path=/dockerdata/Qwen2.5-VL-7B-Instruct
qwen2_5_vl_3b_ckpt_path=/dockerdata/Qwen2.5-VL-3B-Instruct
# Training Arguments
LOCAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
# Log Arguments
export TRANSFORMERS_OFFLINE=1
RUN_NAME=105-arpo_kl-7b-ablation-seed_bench_r1
OUTP_DIR=results
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
export NPU_ENABLE=true
export WANDB_PROJECT=ARPO
export HF_HUB_OFFLINE='False'

cold_start_model_path=/dockerdata/VideoRFT_cold_start_only_video_data_ckpt-910

video_r1_data=''
seed_bench_r1_data='data/rl_training_data.json'
seed_bench_r1_6k_data='data/seed_bench_r1_6k_with_duration.json'
video_r1_seed_bench_r1_44k_data='data/video_r1_34k_seed_bench_r1_10k.json'
video_r1_34k_data='data/video_r1_34k.json'
perception_test_6k_data='data/perception_test_6k.json'
nextqa_34k_data='data/nextqa/train_rl_34k.json'
nextgqa_3k_data='data/nextqa/next-gqa_val_rl_3k3.json'
dynamic_sampling_data='data/seed-bench-r1/dynamic_sampling_data.json'
dynamic_sampling_data_3b='data/seed-bench-r1/dynamic_sampling_data_3b.json'

max_frames=16
min_pixels=50176 # 64 * 28 * 28 
max_pixels=50176 # 64 * 28 * 28

torchrun --nproc_per_node 16 --master_port 6667 \
    rl/grpo_0_23_1.py \
    --deepspeed deepspeed/stage2-offload.json \
    --output_dir results/$WANDB_PROJECT/$RUN_NAME \
    --model_name_or_path $qwen2_5_vl_ckpt_path \
    --dataset_name xxx \
    --jsonl_path $seed_bench_r1_6k_data \
    --reward_funcs video_r1_accuracy video_r1_format \
    --reward_weights 0.9 0.1 \
    --attn_enhanced True \
    --max_completion_length 384 \
    --use_vllm False \
    --max_frames $max_frames \
    --min_pixels $min_pixels \
    --max_pixels $max_pixels \
    --num_generations 8 \
    --learning_rate 1e-6 \
    --loss_type arpo_kl \
    --max_weight 1.7 \
    --select_strategy topk-union \
    --select_level force-all \
    --token_nums_for_avg_frame_score 15 \
    --frame_nums_for_select_valid_frames 3 \
    --token_nums_for_update_per_frame 128 \
    --add_one_to_weights True \
    --epsilon_high 0.28 \
    --beta 0.00 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --logging_steps 1 \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --eval_strategy no \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --save_steps 0.4 \
    --save_only_model true


