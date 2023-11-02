import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed
)
import torch
import time
from transformers import LlamaTokenizer, LlamaConfig
from accelerate import Accelerator
from arguments import (ModelArguments, DataArguments, LoraArguments, ReinforceTrainingArguments as TrainingArguments)
from data.llama_collator import DataCollatorLlama
from policy_modeling import LlamaforPolicyModel
from trainer import PolicyTrainer
import datasets
from log_config import CustomLoggingCallback

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"]="true"


def print_trainable_parameters(model):
    """
    print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _,param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main():
    #torch.cuda.set_device(1)

    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.per_device_train_batch_size =2  # set batch size to 1
    training_args.per_device_eval_batch_size = 1
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

    #Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        #filename=os.path.join(current_log_dir, 'training_info.log'),
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



    train_dataset = datasets.load_from_disk(dataset_path=data_args.train_data)['train']
    #torch.cuda.set_device(1)
    accelerator = Accelerator()
    print(accelerator.device)
    print(torch.cuda.is_available())
    # 获取CUDA设备数量
    print(torch.cuda.device_count())




    #set_seed(training_args.seed)

    config = LlamaConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        # num_hidden_layers=2  # TODO(jax)  for test only !!
    )

    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path)
    tokenizer.pad_token_id = 0
    
    model = LlamaforPolicyModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        config=config,
        # load_in_4bit=True,
        #device_map={"": Accelerator().local_process_index},
    )

    
    lora_config = LoraConfig(
        lora_args,
        #lora_target_modules = model,
        modules_to_save = ["policy_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model.resize_token_embeddings(len(tokenizer))
    
    #print_trainable_parameters(model=model)

    trainer = PolicyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,  # For save checkpoint purpose
        data_collator= DataCollatorLlama(
            tokenizer=tokenizer,
            model=None,
            max_source_length=512,
            gamma=training_args.gamma,
            delta=training_args.delta
        ),
        peft_config=lora_config,
        callbacks=[CustomLoggingCallback]  # TODO(jax) 后续确认一下会不会和我policytrainer里面peft的callback产生影响
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    #trainer.save_model()
    print(type(trainer.model))
    trainer.model.save_pretrained("fine-tuned_dir/", modules_to_save="policy_head")

    #test
    test_input = "caculate 1+1\n<solver>"
    # The inputs should be "Your trajectory\n<solver>"
    model.eval()
    test_input = tokenizer(test_input, return_tensors="pt").to(model.device)  # no padding
    with torch.no_grad():
        print(model.inference(test_input['input_ids']))
        
    #print(output[1])

    



    #TODO(xueyang) resume checkpoint, save final model after merging base_model and lora
    """
    if args.merge_lora:
        merge_llm_with_lora(args.base_model, final_model_path, args.output_dir)
    """

if __name__ == "__main__":
    
    main()

