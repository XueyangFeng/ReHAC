import logging
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from typing import Dict, Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoTokenizer, LlamaPreTrainedModel, LlamaModel, LlamaConfig
from transformers.file_utils import ModelOutput
from torch.distributions import Categorical
logger = logging.getLogger(__name__)


def check_tensor_values(tensor, variable_name):
    # check the value of tensor
    if not torch.all(torch.logical_and(tensor >= 0, tensor <= 1)):
        raise ValueError(f"Tensor '{variable_name}' contains values outside the range [0,1]")

@dataclass
class PolicyOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: torch.FloatTensor = None

@dataclass
class GenerateOutput(ModelOutput):
    logits: torch.int = 0

class LlamaforPolicyModel(LlamaPreTrainedModel):
    def __init__(self, config, alpha):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.action_space_size = 2
        self.policy_head = nn.Linear(config.hidden_size, self.action_space_size, bias=True)
        self.post_init()
        self.alpha = alpha


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, alpha=None, **kwargs):
        if alpha is None:
            raise ValueError("Alpha value must be provided for LlamaforPolicyModel")
        kwargs['alpha'] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        return model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        reward_returns: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        counts: Optional[torch.LongTensor] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.policy_head(hidden_states)
        logits = logits.float()
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        
        behaviour_policy_prob =0.5

        loss = None
        if labels is not None:
            check_tensor_values(labels, 'labels')
            logits_probs = F.softmax(pooled_logits, dim=-1) #batch_size*action_space_size

            labels = labels.unsqueeze(1) #batch_size*1
            with torch.no_grad():
                target_policy_prob = logits_probs.gather(1, labels)  
                important_sampling_ratio = target_policy_prob / behaviour_policy_prob   
            
            distribution = Categorical(logits_probs)
            labels = labels.squeeze(1)
            entropy = distribution.entropy()
            alpha = self.alpha
            clip_threshold = 0.3

            weight = torch.clamp(important_sampling_ratio*counts, 1 - clip_threshold, 1 + clip_threshold)
            loss = -distribution.log_prob(labels)*(reward_returns)*weight - alpha*entropy    
            loss = torch.sum(loss) 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PolicyOutput(
            loss = loss,
            logits = logits
        )

    def inference(self, input_ids, attention_mask):
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs[0]
        
        logits = self.policy_head(hidden_states)
        logits = logits.float()

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            raise

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]        
        logits_probs = F.softmax(pooled_logits, dim=-1)
        action = logits_probs.argmax()
        return action
    
    
