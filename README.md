# MetaToolLearner
The code for the ACM MM2025 paper "MetaToolLearner: Enhancing Cross-Tool Generalization for Dynamic Multimodal Coordination through Meta-Learning"

# RUN
``` bash
python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path {model_path} \
    --finetuning_type lora \
    --dataset_dir data \
    --dataset {data_dir} \
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
    --lora_target query_key_value \
    --output_dir {output_path} \
    --fp16 True \
    --plot_loss True
```