from transformers import TrainerCallback, TrainerState, TrainerControl
import random
from test_data.localgpt import localgpt
import time
import torch
import jsonlines
from data_produce import wikienv, wrappers

import os
work_dir = os.environ["WORK_DIR"]

#learning curve
class TestingCallback(TrainerCallback):
    def __init__(self, alpha, lr, *args, **kwargs):  
        dev_env = wikienv.WikiEnv()
        dev_env = wrappers.HotPotQAWrapper(dev_env , split="dev")  
        dev_env = wrappers.LoggingWrapper(dev_env)
        train_env = wikienv.WikiEnv()
        train_env = wrappers.HotPotQAWrapper(train_env , split="train")  
        train_env = wrappers.LoggingWrapper(train_env)
        self.dev_env = dev_env
        self.train_env = train_env
        self.train_localgpt = localgpt('train')
        self.dev_localgpt = localgpt('dev')
        self.accumulated_backprop_count = 0        
        self.alpha = alpha
        self.lr = lr

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.accumulated_backprop_count += 1
        if self.accumulated_backprop_count % 5 == 0:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            current_epoch = state.epoch
            current_delta = args.delta
            
            if model is not None:
                print("=" * 50 + "TESTING" + "=" * 50)
                model.eval()

                test_model(alpha=self.alpha, model=model, epoch=current_epoch, delta=current_delta, tokenizer=tokenizer, dev_env=self.dev_env, train_env=self.train_env, train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt, lr = self.lr)
                model.train()
                print("=" * 50 + "TRAINING" + "=" * 50)
                
            else:
                raise ValueError("Model is None")


    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        current_delta = args.delta

        if model is not None:
            model.eval()
            test_model(alpha=self.alpha, model=model, epoch=0, delta=current_delta, tokenizer=tokenizer, dev_env=self.dev_env, train_env=self.train_env, train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt, lr = self.lr)
            model.train()
        else:
            raise ValueError("Model is None")




def localthink(idx, tokenizer, model, env, localgpt, to_print=False):
    question = env.reset(idx=idx)

    output = ""
    output += question + "\n"
    
    solver_token ="<solver>"
    for i in range(1, 10):
        input = output + solver_token
        test_input = tokenizer(input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            solver_id = model.inference(test_input['input_ids'], test_input['attention_mask']).item()
        if (1 == solver_id):
            solver = "<solver> human\n"
        elif (0 == solver_id):
            solver = "<solver> agent\n"
        info = localgpt.get_response(idx, output + solver)
        step_str = info['trajectory']
        output += solver + step_str
        if to_print:
            print(step_str)
        if "f1" in info:
            break
    return info, output


def test_model(alpha, model, epoch, delta, dev_env, train_env, tokenizer, train_localgpt, dev_localgpt, lr):

    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)


    rs = []
    infos = []
    old_time = time.time()
    with jsonlines.open(work_dir + f"/reward_data/delta{delta}_alpha{alpha}_lr{lr}_dev100_log/epoch{epoch}.jsonl", "a") as jsonl_file:
        for i in idxs[0:100]: 
            info, trajectory = localthink(i, tokenizer=tokenizer, model=model, env=dev_env, localgpt=dev_localgpt, to_print=False)
            rs.append(info['em'])
            infos.append(info)
            # Write trajectory into jsonl
            jsonl_file.write({"trajectory": trajectory, "em": info['em'], "f1": info['f1']})

    idxs = list(range(90447)) # train data set
    random.Random(233).shuffle(idxs) 
    all_list = [5, 18, 37, 40, 13, 19, 27, 32, 42, 50, 51, 53, 54, 58, 62, 64, 65, 71, 72, 74, 75, 78, 80, 84, 85, 87, 91, 93, 94, 95, 97, 98, 100, 102, 106, 109, 112, 116, 118, 119, 121, 122, 124, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249]
    with jsonlines.open(work_dir + f"/reward_data/delta{delta}_alpha{alpha}_lr{lr}_train160_log/epoch{epoch}.jsonl", "a") as jsonl_file:
        for j in all_list: 
            i = idxs[j]
            info, trajectory = localthink(i, tokenizer=tokenizer, model=model, env= train_env, localgpt=train_localgpt, to_print=False)
            rs.append(info['em'])
            infos.append(info)
            # Write trajectory into jsonl
            jsonl_file.write({"trajectory": trajectory, "em": info['em'], "f1": info['f1']})     
