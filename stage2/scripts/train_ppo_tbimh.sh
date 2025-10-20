#!/bin/bash
set -e
set -x

# ==================== 硬件与环境 ====================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD:$PYTHONPATH"
# export TRANSFORMERS_OFFLINE=1


export DATA_DIR="stage2/data"     
export MODEL_DIR="."               

# ==================== 模型配置 ====================
VISION_TOWER="openai/clip-vit-large-patch14-336"


BASE_MODEL_NAME="./checkpoints/tbimh_step3_vqa_full-13b"


POLICY_LORA="TBIMH-PolicyInit-13b-lora/lora_default"

RM_LORA="TBIMH-RM-13b-lora/checkpoint-200"

# ==================== 数据配置（PPO训练语料） ====================
PPO_DATA="tbimh_ppo_mix.jsonl"
IMAGE_ROOT="tbimh_images"
IMAGE2CAP="$DATA_DIR/image_to_caption.json"
REWARD_PROMPT_FILE="./prompts/tbimh_reward_prompt.txt"

# ==================== 保存配置 ====================
MODEL_NAME="TBIMH-RL-PPO-13b-lora"
OUTPUT_DIR="$MODEL_DIR/$MODEL_NAME"

# ==================== 训练超参（先小步稳定） ====================
LEARNING_RATE=3e-5
KL_COEF=0.1
TOTAL_EPOCHS=4

ROLLOUT_BATCH_SIZE=256               # 总采样 batch
STEP_BATCH_SIZE=128                  # 总更新 batch
ROLLOUT_PER_DEVICE_BATCH_SIZE=16     # 每卡采样 batch
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=8 # 每卡RM前向 batch
STEP_PER_DEVICE_BATCH_SIZE=8         # 每卡更新 batch
NOPTEPOCHS=2

INCOMPLETE_RESPONSE=-8.0
LENGTH_BONUS=-10.0
CORRECT_BONUS=2.0

# ==================== 启动 PPO ====================
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=$GPUS_PER_NODE \
  train/finetune_lora_ppo.py \
  --do_train \
  --seed 42 \
  --step_batch_size $STEP_BATCH_SIZE \
  --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
  --rollout_batch_size $ROLLOUT_BATCH_SIZE \
  --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
  --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
  --base_model_name "$BASE_MODEL_NAME" \
  --policy_model_name_or_path "$MODEL_DIR/$POLICY_LORA" \
  --reward_model_name_or_path "$MODEL_DIR/$RM_LORA" \
  --learning_rate $LEARNING_RATE \
  --init_value_with_reward True \
  --warmup_steps 5 \
  --dataset_path "$DATA_DIR/$PPO_DATA" \
  --train_splits "train" \
  --output_dir "$OUTPUT_DIR" \
  --total_epochs $TOTAL_EPOCHS \
  --group_by_length False \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 10 \
  --save_total_limit 100000 \
  --weight_decay 0.0 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "tensorboard" \
  --ddp_backend "nccl" \
  --bf16 True \
  --penalty_reward_value $INCOMPLETE_RESPONSE \
  --length_bonus_score $LENGTH_BONUS \
  --correct_bonus_score $CORRECT_BONUS \
  --relative_stop_token_penalty True \
  --penalize_no_stop_token True \
  --ddp_find_unused_parameters False \
  --resume_from_training True \
  --kl_coef $KL_COEF \
  --max_grad_norm 1.0 \
  --whitening_async_stats "full_batch" \
  --clean_tokens_after_eos True \
  --temperature 1.0 \
  --whiten_rewards False \
  --model_max_length 2048 \
  --query_len 128 \
  --response_len 896 \
  --noptepochs $NOPTEPOCHS \
  --image_folder "$DATA_DIR/$IMAGE_ROOT" \
  --vision_tower "$VISION_TOWER" \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --freeze_mm_mlp_adapter True \
  --reward_prompt_file "$REWARD_PROMPT_FILE" \
  --image_to_caption_file "$IMAGE2CAP" \
  --image_aspect_ratio 'pad'
