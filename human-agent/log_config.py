import os
import time
from transformers import TrainerCallback
from arguments import (ModelArguments, DataArguments, LoraArguments, ReinforceTrainingArguments as TrainingArguments)

class CustomLoggingCallback(TrainerCallback):
    
    def __init__(self):
        # 初始化时获取时间戳
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f".log/{self.timestamp}"
        
        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        # 在训练开始时，保存超参数和配置信息
        
        with open(f"{self.log_dir}/config.txt", "w") as f:
            f.write(str(args))


    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            last_log = state.log_history[-1]
            current_loss = last_log.get('loss', None)
        else:
            current_loss = None

        #print(state)

        all_losses = [log.get('loss', 0.0) for log in state.log_history]
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0

        with open(f"{self.log_dir}/train_loss.txt", "a") as f:
            for log in state.log_history:
                f.write(f"{log}\n")
            state.log_history = []
            f.write(f"Epoch {state.epoch} - Avg Loss: {avg_loss}\n")





          
