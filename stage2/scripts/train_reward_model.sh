#!/bin/bash
set -e
set -x

# ====== 硬件与环境 ======
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD:$PYTHONPATH"

# ====== 路径（按你的仓库调整）======
export DATA_DIR="stage2/data"           # 例如: /data/tbimh_rlhf
export MODEL_DIR="."        # 例如: /models/tbimh
export LOG_DIR="stage2/logs"            # 可选

# ====== 模型配置 ======
VISION_TOWER="openai/clip-vit-large-patch14-336"
# !!! 用 Stage 1（第一阶段第二步）产出的检查点作为RM背后的骨干模型 !!!
LM_MODEL_NAME="./checkpoints/tbimh_step3_vqa_full-13b"             # 例如: checkpoints/tbimh_step2_instruct-13b

# ====== 数据配置 ======
# 偏好数据 = 医生选择 A/B 哪个回答更好（pairwise preference）
PREFERENCE_DATA="tbimh_preference_train.jsonl"       # 训练
PREFERENCE_EVAL="tbimh_preference_eval.jsonl"        # 验证（可用训练集子集先替代）
# 影像根目录（CTA等），不是 COCO 路径
IMAGE_ROOT="tbimh_images"                             # 例如: /data/tbimh/images

# ====== 保存配置 ======
MODEL_NAME="TBIMH-RM-13b-lora"
OUTPUT_DIR="$MODEL_DIR/$MODEL_NAME"

# ====== 训练超参（建议从小步稳定开始）======
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=4
GRAD_ACCUMULATION=1
EVAL_SIZE=200

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=$GPUS_PER_NODE \
  train/finetune_lora_rm.py \
  --do_train \
  --do_eval \
  --seed 42 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME \
  --image_folder $DATA_DIR/$IMAGE_ROOT \
  --vision_tower $VISION_TOWER \
  --learning_rate $LEARNING_RATE \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --freeze_mm_mlp_adapter True \
  --model_max_length 2048 \
  --query_len 1280 \
  --response_len 768 \
  --dataset_path $DATA_DIR/$PREFERENCE_DATA \
  --eval_dataset_path $DATA_DIR/$PREFERENCE_EVAL \
  --dataset_name "none" \
  --eval_dataset_name "none" \
  --eval_size $EVAL_SIZE \
  --bits 16 \
  --lora_r 64 \
  --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs $NUM_EPOCHS \
  --group_by_length False \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_total_limit 10 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 5 \
  --report_to "tensorboard" \
  --ddp_backend "nccl" \
  --bf16 True \
  --ddp_find_unused_parameters False \
  --resume_from_training True \
  --reward_prompt_file "./prompts/tbimh_reward_prompt.txt" \
  --image_to_caption_file "$DATA_DIR/image_to_caption.json" \
  --image_aspect_ratio 'pad'
