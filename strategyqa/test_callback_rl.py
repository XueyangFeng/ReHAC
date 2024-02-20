from transformers import TrainerCallback, TrainerState, TrainerControl
import random
from test_data.localgpt import localgpt
import time
import torch
import jsonlines
from data_produce import wikienv, wrappers
import os
work_dir = os.environ["WORK_DIR"]

class TestingCallback(TrainerCallback):
    def __init__(self, alpha, lr, *args, **kwargs):  
        dev_env = wikienv.WikiEnv()
        dev_env = wrappers.StrategyQAWrapper(dev_env , split="test")  # TODO(jax) 用test数据
        dev_env = wrappers.LoggingWrapper(dev_env)
        train_env = wikienv.WikiEnv()
        train_env = wrappers.StrategyQAWrapper(train_env , split="train")  # TODO(jax) 用train数据
        train_env = wrappers.LoggingWrapper(train_env)
        self.dev_env = dev_env
        self.train_env = train_env
        self.train_localgpt = localgpt('train')
        self.dev_localgpt = localgpt('dev')
        self.accumulated_backprop_count = 0
        self.alpha = alpha
        self.lr = lr

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 每次反向传播后，增加计数
        self.accumulated_backprop_count += 1

        # 检查是否达到了 6 次反向传播
        if self.accumulated_backprop_count % 5 == 0:
            # 每 6 次反向传播后执行的操作
            # 获取当前训练过程中的模型
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            current_epoch = state.epoch
            current_delta = args.delta
            
            if model is not None:
                print("=" * 50 + "TESTING" + "=" * 50)
                model.eval()
                # 在这里使用模型进行测试或其他操作
                test_model(alpha=self.alpha, model=model, epoch=current_epoch, delta=current_delta, tokenizer=tokenizer, dev_env=self.dev_env, train_env=self.train_env, train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt, lr=self.lr)
                model.train()
                print("=" * 50 + "TRAINING" + "=" * 50)
                
            else:
                # 处理模型未提供的情况
                raise ValueError("Model is None")


    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 获取模型和分词器
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        current_delta = args.delta

        if model is not None:
            # 切换到评估模式
            model.eval()

            # 执行测试
            test_model(alpha=self.alpha, model=model, epoch=0, delta=current_delta, tokenizer=tokenizer, dev_env=self.dev_env, train_env=self.train_env, train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt, lr=self.lr)

            # 切换回训练模式
            model.train()
        else:
            # 模型未提供的处理
            raise ValueError("Model is None")

    """
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 获取当前训练过程中的模型
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        current_epoch = state.epoch
        current_delta = args.delta
        
        if model is not None:
            model.eval()
            # 在这里使用模型进行测试或其他操作
            test_model(model=model, epoch=current_epoch, delta=current_delta, tokenizer=tokenizer, dev_env=self.dev_env, train_env=self.train_env, train_localgpt=self.train_localgpt, dev_localgpt=self.dev_localgpt)
            model.train()
            
        else:
            # 处理模型未提供的情况
            raise ValueError("Model is None")
    """



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

    idxs = list(range(300))
    random.Random(233).shuffle(idxs) 

    rs = []
    infos = []

    with jsonlines.open(work_dir + f"/reward_data/delta{delta}_alpha{alpha}_lr{lr}_dev100_log/epoch{epoch}.jsonl", "a") as jsonl_file:
        for i in idxs[0:100]: #except 20
            info, trajectory = localthink(i, tokenizer=tokenizer, model=model, env=dev_env, localgpt=dev_localgpt, to_print=False)
            rs.append(info['em'])
            infos.append(info)
            # Write trajectory into jsonl
            jsonl_file.write({"trajectory": trajectory, "em": info['em'], "f1": info['f1']})
    
    idxs = list(range(500)) # train data set
    random.Random(233).shuffle(idxs) 

    rs = []
    infos = []
   
    all_list = list(range(0, 169)) + list(range(170, 251))  # 169 exists some bug

    #idxs = list(range(90447)) # train data set
    #random.Random(233).shuffle(idxs) 
    with jsonlines.open(work_dir + f"/reward_data/delta{delta}_alpha{alpha}_lr{lr}_train250_log/epoch{epoch}.jsonl", "a") as jsonl_file:
        for j in all_list: #except 20
            i = idxs[j]
            info, trajectory = localthink(i, tokenizer=tokenizer, model=model, env= train_env, localgpt=train_localgpt, to_print=False)
            rs.append(info['em'])
            infos.append(info)
            # Write trajectory into jsonl
            jsonl_file.write({"trajectory": trajectory, "em": info['em'], "f1": info['f1']})     
