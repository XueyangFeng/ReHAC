export CUDA_VISIBLE_DEVICES=0
python run.py \
    --output_dir /home/fengxueyang/rl/jax_hc/Gradient-based-Human-Agent-Collaboration/human-agent/fine-tuned_dir \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train True \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --logging_strategy steps \
    --logging_steps 2 \
    --save_strategy no \
    --seed 42 \
    --report_to none \
    --model_name_or_path /home/fengxueyang/rl/gradient_human/pretrain_model/llama_hf \
    --train_data /home/fengxueyang/rl/jax_hc/Gradient-based-Human-Agent-Collaboration/human-agent/data/data_solver_hf_100 \
    --max_trajectory_length 16 \
    --gamma 0.9 \
    --delta 0.1