import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/root/autodl-tmp/pretrain_dir/llama_hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    tokenizers: Optional[str] = field(
        default="/root/autodl-tmp/pretrain_dir/llama_hf", metadata={"help": "Path to pretrain tokenizer"}
    )
    use_fast_tokenizer: bool = field(default=True)



@dataclass
class DataArguments:
    train_data: Optional[str] = field(
        default="./data/data_solver_hf", metadata={"help": "Path to train data"}
    )

    max_trajectory_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input trajectory length after tokenization. trajectorys longer "
                    "than this will be truncated, trajectorys shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000, metadata={"help": "the max number of examples for each dataset"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class LoraArguments:
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default=None)

@dataclass
class ReinforceTrainingArguments(TrainingArguments):
    #negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    #fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    #sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    gamma: float = field(default=1.0, metadata={"help": "return decay rate"})
    delta: float = field(default=0, metadata={"help": "human decay value  real_return = return - delta * human_times"})
    logging_steps: int = field(default=20)
    per_device_train_batch_size: int = field(default=16)
    output_dir: str = field(default="fine-tuned_dir")
    alpha: float = field(default=0.1, metadata={"help": "Alpha value for training"})
