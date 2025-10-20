#!/bin/bash
set -e
set -x

# ====== 硬件与环境 ======
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD:$PYTHONPATH"

# ====== 路径（与第一步保持一致）======
export DATA_DIR="stage2/data"
export MODEL_DIR="stage2/models"
export LOG_DIR="stage2/logs"

# ====== 模型配置 ======
VISION_TOWER="openai/clip-vit-large-patch14-336"
# 用 Stage-1 Step3 产出的 SFT/VQA 模型作为 RL policy 的起点
LM_MODEL_NAME="./checkpoints/tbimh_step3_vqa_full-13b"   # 已是完整路径，就不要再拼 MODEL_DIR 了

# ====== 数据配置（SFT 格式，而不是偏好数据）======
SFT_DATA="tbimh_sft_init.json"     # 指令跟随数据（可复用你 Step2/Step3 的指令数据）
IMAGE_ROOT="tbimh_images"

# ====== 保存配置 ======
MODEL_NAME="TBIMH-RLHF-13b-policy-init"
OUTPUT_DIR="$MODEL_DIR/$MODEL_NAME"

# ====== 训练超参（轻量）======
NUM_EPOCHS=1
LEARNING_RATE=1e-4
BATCH_SIZE=8
GRAD_ACCUMULATION=2

deepspeed \
  finetune_lora_sft_ds.py \
  --deepspeed scripts/zero2.json \
  --do_train \
  --do_eval \
  --seed 42 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --model_name_or_path "$LM_MODEL_NAME" \
  --image_folder "$DATA_DIR/$IMAGE_ROOT" \
  --vision_tower $VISION_TOWER \
  --learning_rate $LEARNING_RATE \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --freeze_mm_mlp_adapter True \
  --query_len 1152 \
  --response_len 640 \
  --dataset "$DATA_DIR/$SFT_DATA" \
  --dataset_format "v1" \
  --eval_size 200 \
  --bits 16 \
  --lora_r 64 \
  --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs $NUM_EPOCHS \
  --group_by_length False \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --save_strategy "steps" \
  --save_steps 1000000 \
  --save_total_limit 1 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --report_to "tensorboard" \
  --ddp_backend "nccl" \
  --bf16 True \
  --ddp_find_unused_parameters False \
  --resume_from_training True \
  --image_aspect_ratio 'pad'
