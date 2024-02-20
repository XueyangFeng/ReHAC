import logging
import os
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

    set_seed(training_args.seed)
    
    # Load from dataset file
    # train_dataset = datasets.load_from_disk(dataset_path=data_args.train_data)['train']
    train_dataset = datasets.load_dataset("json", data_files=data_args.train_data, cache_dir="./json2hf_dataset")['train']
    
    config = LlamaConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    config.pad_token_id = 0

    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.pad_token_id = 0
    

    model = LlamaforPolicyModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
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
        trainer.save_model()  
        
        metrics = train_result.metrics
        metrics["train_samples"] = train_dataset.num_rows
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()

