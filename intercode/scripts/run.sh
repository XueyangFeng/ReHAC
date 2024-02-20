Data=100
DELTA=0.08
EPOCH=10
LR=5e-05
ALPHA=0.0
BS=64

export WORK_DIR="your/code/directory"

MODEL_CACHE_DIR="your/llama_model/directory"
TRAIN_DATA_DIR="your/processed_training_data/directory"

OUTPUT_DIR="./fine-tuned_dir_ppo_is/train_exhaust_reward${Data}/delta${DELTA}_epoch${EPOCH}_alpha${ALPHA}_batchsize${BS}_lr${LR}/"
LOG_DIR="./log/train_exhaust_${Data}/delta${DELTA}_epoch${EPOCH}_alpha${ALPHA}/"
TRAIN_REWARD_DIR="./reward_data/ppo${DELTA}_train${Data}_log_alpha${ALPHA}_lr${LR}_argmax/"
DEV_REWARD_DIR="./reward_data/ppo${DELTA}_dev100_log_alpha${ALPHA}_lr${LR}_argmax/"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
mkdir -p $TRAIN_REWARD_DIR
mkdir -p $DEV_REWARD_DIR

python run_cls.py \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train True \
    --per_device_train_batch_size 1 \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --logging_strategy steps \
    --gradient_accumulation_steps $BS \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --seed 42 \
    --alpha $ALPHA \
    --report_to none \
    --model_name_or_path $MODEL_CACHE_DIR \
    --train_data $TRAIN_DATA_DIR \
    --max_trajectory_length 1024 \
    --delta $DELTA \
    --logging_dir $LOG_DIR
