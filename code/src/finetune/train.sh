#!/bin/bash

# 設定環境變數
export CUDA_VISIBLE_DEVICES=0
MODEL="huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"

INPUT_FILE="data/example_data.jsonl"
PREPROCESSED_DATAPATH="data/processed_data.jsonl"
OUTPUT_DIR="finetuned_models/Qwen2.5-14B-finetuned"

mkdir -p $OUTPUT_DIR

python preprocess.py \
    --input_file $INPUT_FILE \
    --output_file $PREPROCESSED_DATAPATH \

python qlora.py \
    --model_name_or_path $MODEL \
    --dataset $PREPROCESSED_DATAPATH \
    --dataset_format "input-output" \
    --output_dir $OUTPUT_DIR \
    --do_train True \
    --do_eval True \
    --eval_dataset_size 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2   \
    --gradient_accumulation_steps 8 \
    --max_steps 500 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --source_max_len 256 \
    --target_max_len 512 \
    --bits 4 \
    --quant_type "nf4" \
    --double_quant True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --gradient_checkpointing True \
    --bf16 True \
    --optim "paged_adamw_8bit" \
    --report_to "none" \
    --seed 42
