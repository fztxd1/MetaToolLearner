CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /root/autodl-tmp/model/chinese-alpaca-2-7b \
    --finetuning_type lora \
    --template llama2_zh \
    --dataset_dir data \
    --dataset toolaugmented_train_WT \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 50 \
    --warmup_steps 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir /root/autodl-tmp/test02/LLaMa_7b_sft_WT_3epoch \
    --fp16 True \
    --plot_loss True