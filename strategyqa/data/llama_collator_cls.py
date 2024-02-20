from transformers.data.data_collator import *
import torch

@dataclass
class DataCollatorLlama:
    
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any]
    max_source_length: Optional[int] = 1024
    padding: str = "longest"
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
   
    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        trajs = []
        labels = []
        rewards = []
        counts = []
        
        for instance in batch:
            source = instance['state']
            tokenized_source = self.tokenizer(source)['input_ids']
            if len(tokenized_source) <= self.max_source_length:  # pass exceed max_source_length
                trajs.append(source)
                labels.append(instance['action'])
                rewards.append(instance['reward'])
                counts.append(instance['count'])

        
        
        outputs = self.tokenizer(trajs, max_length=self.max_source_length, padding=self.padding, return_tensors=self.return_tensors, 
                                 truncation=True, pad_to_multiple_of=self.pad_to_multiple_of)
        outputs["labels"] = torch.Tensor(labels).long()
        outputs["reward_returns"] = torch.Tensor(rewards)
        outputs['counts'] = torch.Tensor(counts).long()
        ##############################
        # TODO(jax) if no instance is less than max_source_length there will be a wrong
        ##############################
        return outputs