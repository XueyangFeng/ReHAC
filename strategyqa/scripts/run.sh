Data=250
DELTA=0.08
ALPHA=0.1
LR=1e-05
BS=64
EPOCH=10

export WORK_DIR="your/code/directory"

MODEL_CACHE_DIR="your/llama_model/directory"
TRAIN_DATA_DIR="your/processed_training_data/directory"

OUTPUT_DIR=${WORK_DIR}"/ppo_${Data}/delta${DELTA}_alpha${ALPHA}_bs${BS}_lr${LR}_epoch${EPOCH}/"
TRAIN_REWARD_DIR=${WORK_DIR}"/reward_data/delta${DELTA}_alpha${ALPHA}_lr${LR}_train${Data}_log/"
DEV_REWARD_DIR=${WORK_DIR}"/reward_data/delta${DELTA}_alpha${ALPHA}_lr${LR}_dev100_log/"

mkdir -p $OUTPUT_DIR
mkdir -p $TRAIN_REWARD_DIR
mkdir -p $DEV_REWARD_DIR

python ${WORK_DIR}/run_cls.py \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train True \
    --per_device_train_batch_size 1 \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --logging_strategy steps \
    --gradient_accumulation_steps $BS \
    --logging_steps 20 \
    --save_strategy no \
    --bf16 True \
    --seed 42 \
    --alpha $ALPHA \
    --report_to none \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --cache_dir $MODEL_CACHE_DIR \
    --train_data $TRAIN_DATA_DIR \
    --max_trajectory_length 2048 \
    --delta $DELTA