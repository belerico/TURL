CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch --nproc-per-node=1 ./run_hybrid_table_lm_finetuning.py \
    --output_dir='./output' \
    --model_type=hybrid \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --data_dir="~/turl-data" \
    --evaluate_during_training \
    --mlm \
    --mlm_probability=0.2 \
    --ent_mlm_probability=0.6 \
    --mall_probability=0.7 \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --num_train_epochs=1 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --max_entity_candidate=10000 \
    --config_name=src/configs/table-base-config_v2.json \
    --save_steps=40 \
    --logging_steps=40 \
    --use_cand \
    --fp16 \
    --exclusive_ent=0 \
    --random_sample \
    --do_eval \
    --eval_all_checkpoints \
    --dry_run \