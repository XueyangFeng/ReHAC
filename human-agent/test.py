from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed
)
import torch
from accelerate import Accelerator
from policy_modeling import LlamaforPolicyModel
from arguments import (ModelArguments, DataArguments, LoraArguments, ReinforceTrainingArguments as TrainingArguments)
from transformers import LlamaTokenizer, LlamaConfig

parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

config = LlamaConfig.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name_or_path,
)

model = LlamaforPolicyModel.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name_or_path,
    config=config,
    #load_in_4bit=True,
    #device_map={"": Accelerator().local_process_index},
)



model = PeftModel.from_pretrained(model, "./fine-tuned_dir")
print("The weight of output header and lora has been initialized")

tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path)

############  test
test_input = "caculate 1+1\n<solver>"
# The inputs should be "Your trajectory\n<solver>"
model.eval()
test_input = tokenizer(test_input, return_tensors="pt").to(model.device)  # no padding
with torch.no_grad():
    print(model.inference(test_input['input_ids']))
