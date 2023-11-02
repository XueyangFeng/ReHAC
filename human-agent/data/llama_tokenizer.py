from datasets import (load_from_disk)
from transformers import AutoTokenizer
from llama_collator import DataCollatorLlama




dataset = load_from_disk('/home/fengxueyang/rl/jaxchen/data/data_solver_hf')['train']

trajs = []
for i in range(dataset.num_rows):
     traj = dataset[i]['trajectory']
     trajs.append(traj)

### solver>: 2929, 369, 29958
### agent: 10823
### human: 5199

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="hf_model/")
tokenizer = AutoTokenizer.from_pretrained("/home/fengxueyang/rl/gradient_human/pretrain_model/llama_hf")
tokenizer.pad_token_id = 0

collator_fn = DataCollatorLlama(
    tokenizer=tokenizer,
    model=None,
    max_source_length=512,
)

result = collator_fn.__call__(trajs[0])
print(type(result))
#result.data.
a = result["input_ids"]
print(a)
#b = result["labels"]
#print(result["input_ids"])
#print(result.data())


    
    
