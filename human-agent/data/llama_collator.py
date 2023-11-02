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
    gamma: float = 1.0  # return decay value
    delta: float = 0.  # human decay value  real_return = return - delta * human_times
    
    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        trajs = []
        f1_scores = []
        for instance in batch:
            trajs.append(instance['trajectory'])
            f1_scores.append(instance['f1'])
            #trajs.append(instance[0])
            #f1_scores.append(instance[2])
        
        # trajs = batch['trajectory']
        # f1_scores = batch['f1']
        
        outputs = self.tokenizer(trajs, max_length=self.max_source_length, padding=self.padding, return_tensors=self.return_tensors, 
                                 truncation=True, pad_to_multiple_of=self.pad_to_multiple_of)
        
        #print(len(outputs))
        output = outputs['input_ids']
        #outputs['input_ids'] = outputs

        batch_size = output.shape[0]
        #print(output.shape[0])
        #print(output.shape[1])
        label_masks = torch.zeros(output.shape)
        labels = torch.zeros(output.shape)
        prediction_idx = torch.nonzero(torch.eq(output, 29958))
        for sub_idx in prediction_idx:
            row_idx, col_idx = sub_idx
            if output[row_idx, col_idx - 1] == 369 and output[row_idx, col_idx - 2] == 2929:
                if output[row_idx, col_idx + 1] == 10823:
                    label_masks[row_idx, col_idx] = 1
                    labels[row_idx, col_idx] = 0  # agent
                if output[row_idx, col_idx + 1] == 5199:
                    label_masks[row_idx, col_idx] = 1
                    labels[row_idx, col_idx] = 1  # human
        outputs["label_masks"] = label_masks
        outputs["labels"] = labels

        ##########################################
        # TODO(chen) move it into trainer module
        reward_returns = torch.zeros(label_masks.shape)
        human_decays = torch.zeros(label_masks.shape)
        
        for i in range(batch_size):
            f1_score = f1_scores[i]
            label_mask = label_masks[i]
            reward_return = reward_returns[i]
            human_decay = human_decays[i]
            
            prediction_idx = torch.nonzero(label_mask)
            prediction_len = prediction_idx.shape[0]
            for idx in range(prediction_len):
                sub_idx_human = idx
                sub_idx_return = prediction_len - idx - 1
                if idx == 0:
                    human_decay[prediction_idx[sub_idx_human]] = self.delta
                    reward_return[prediction_idx[sub_idx_return]] = f1_score
                else:
                    human_decay[prediction_idx[sub_idx_human]] = self.delta + human_decay[prediction_idx[sub_idx_human - 1]]
                    reward_return[prediction_idx[sub_idx_return]] = self.gamma * reward_return[prediction_idx[sub_idx_return + 1]]
            
        outputs["reward_returns"] = reward_returns
        outputs["human_decays"] = human_decays


        # TODO(chen) add prepare deocder_input_ids
        return outputs