import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # TODO(jax) write in bash document
from pathlib import Path
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed
)
from test_callback_rl import TestingCallback
import torch
import time
from transformers import LlamaTokenizer, LlamaConfig
from arguments import (ModelArguments, DataArguments, LoraArguments, ReinforceTrainingArguments as TrainingArguments)
from data.llama_collator_cls import DataCollatorLlama
from policy_modeling_cls import LlamaforPolicyModel
from trainer import PolicyTrainer
import datasets
# from log_config import CustomLoggingCallback

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    alpha_value = training_args.alpha
    lr_value = training_args.learning_rate

    
    training_args.remove_unused_columns = False  # prohibit removing columns
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    current_log_dir = f".log/{timestamp}"

    

    testingCallback = TestingCallback(alpha=alpha_value, lr=lr_value)

    #Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)   # TODO(jax) 随机数种子 data种子 不要注释
    
    # Load from dataset file
    # train_dataset = datasets.load_from_disk(dataset_path=data_args.train_data)['train']
    
    # Load from jsonl file. Here data_args.train_data is the path to the jsonl file
    train_dataset = datasets.load_dataset("json", data_files=data_args.train_data, cache_dir="./json2hf_dataset")['train']
    
    config = LlamaConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        # num_hidden_layers=2  # TODO(jax)  remove it ! for test only !!
    )
    config.pad_token_id = 0

    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.pad_token_id = 0
    

    model = LlamaforPolicyModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        alpha=alpha_value,
        torch_dtype=torch.bfloat16,
        # load_in_4bit=True,
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        target_modules=lora_args.lora_target_modules,  # llama default ["q_proj", "v_proj"]
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        modules_to_save=["policy_head"],  # auto unfreeze policy_head
        task_type="CAUSAL_LM",  # TOKEN_CLS
    )
    # model.resize_token_embeddings(len(tokenizer))
    
    trainer = PolicyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,  # For save checkpoint purpose
        data_collator= DataCollatorLlama(
            tokenizer=tokenizer,
            model=None,
            max_source_length=data_args.max_trajectory_length,
        ),
        peft_config=lora_config,
        callbacks=[testingCallback]  # TODO(jax)
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()

        #trainer.model.save_pretrained("fine-tuned_dir/", modules_to_save="policy_head")

        trainer.save_model()  # TODO(jax) check peft saving
        
        metrics = train_result.metrics
        metrics["train_samples"] = train_dataset.num_rows
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # print(type(trainer.model))  # TODO(jax)
        # trainer.model.save_pretrained("fine-tuned_dir/", modules_to_save="policy_head")

    #test
    # test_input = "caculate 1+1\n<solver>"
    # # The inputs should be "Your trajectory\n<solver>"
    # model.eval()
    # test_input = tokenizer(test_input, return_tensors="pt").to(model.device)  # no padding
    # with torch.no_grad():
    #     print(model.inference(test_input['input_ids']))


    #TODO(xueyang) resume checkpoint, save final model after merging base_model and lora
    """
    if args.merge_lora:
        merge_llm_with_lora(args.base_model, final_model_path, args.output_dir)
    """

if __name__ == "__main__":
    main()

