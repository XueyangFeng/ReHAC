from transformers import TrainerCallback, TrainerState, TrainerControl
import random
from test_data.localgpt import localgpt
from test_advantage import PolicyTester, AgentTester
import time
import torch
import jsonlines
import os
work_dir = os.environ["WORK_DIR"]


train_question_dir = work_dir + "/data/sql/train_question.jsonl"
test_question_dir = work_dir + "/data/sql/dev_question.jsonl"

#learning curve
class TestingCallback(TrainerCallback):
    def __init__(self, alpha, lr, *args, **kwargs):  
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
                test_model(alpha=self.alpha, model=model, epoch=current_epoch, delta=current_delta, tokenizer=tokenizer,  train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt, lr = self.lr)
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
            test_model(alpha=self.alpha, model=model, epoch=0, delta=current_delta, tokenizer=tokenizer,  train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt, lr = self.lr)
            model.train()
        else:
            raise ValueError("Model is None")


    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        current_delta = args.delta

        if model is not None:
            model.eval()
            test_model(alpha=self.alpha, model=model, epoch=10, delta=current_delta, tokenizer=tokenizer,  train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt, lr = self.lr)
            model.train()
        else:
            raise ValueError("Model is None")


def localthink(idx, question, tokenizer, model, localgpt, to_print=False):

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


def test_model(alpha, model, epoch, delta, tokenizer, train_localgpt, dev_localgpt, lr):
    rs = []
    infos = []
    old_time = time.time()
    with jsonlines.open(work_dir + f"/reward_data/ppo{delta}_dev100_log_alpha{alpha}_lr{lr}_argmax/epoch{epoch}.jsonl", "a") as jsonl_file:
        with jsonlines.open(test_question_dir,"r") as f:
            for entry in f:
                idx = entry['idx']
                question = entry['question']
                info, trajectory = localthink(idx, question, tokenizer=tokenizer, model=model, localgpt=dev_localgpt, to_print=False)
                infos.append(info)
                jsonl_file.write({"trajectory": trajectory, "em": info['em'], "f1": info['f1']})
        
    with jsonlines.open(work_dir + f"/reward_data/ppo{delta}_train100_log_alpha{alpha}_lr{lr}_argmax/epoch{epoch}.jsonl", "a") as jsonl_file:
        with jsonlines.open(train_question_dir,"r") as f:
            for entry in f:
                idx = entry['idx']
                question = entry['question']
                info, trajectory = localthink(idx, question, tokenizer=tokenizer, model=model, localgpt=train_localgpt, to_print=False)
                infos.append(info)
                jsonl_file.write({"trajectory": trajectory, "em": info['em'], "f1": info['f1']})
  
