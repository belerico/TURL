MODE=0
CUDA_VISIBLE_DEVICES="0" python run_table_EL_finetuning.py \
    --output_dir=~/projects/TURL/output \
    --model_name_or_path=~/projects/TURL/turl-pretrain-ddp-warmup/checkpoint-last \
    --model_type=EL \
    --do_train \
    --data_dir="~/turl-data" \
    --per_gpu_train_batch_size=10 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-5 \
    --num_train_epochs=10 \
    --save_total_limit=5 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --config_name=~/projects/TURL/src/configs/table-base-config_v2.json \
    --save_steps=1000 \
    --logging_steps=500 \
    --allow_tf32 \
    --mode=$MODE
    # --evaluate_during_training \
    # --per_gpu_eval_batch_size=10 \

