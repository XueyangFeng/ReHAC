import logging
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from typing import Dict, Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoTokenizer, LlamaPreTrainedModel, LlamaModel
from transformers.file_utils import ModelOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
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
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.action_space_size = 2
        self.policy_head = nn.Linear(config.hidden_size, self.action_space_size, bias=False)
        #print("fuck")
        #print(self.policy_head.weight.shape)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.policy_head

    """
    def set_output_embeddings(self, new_embeddings):
        self.policy_head = new_embeddings
    """
    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        reward_returns: Optional[torch.LongTensor] = None,
        human_decays: Optional[torch.LongTensor] = None,
        label_masks: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #print("hello")
        #print(input_ids.shape)

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
        if self.config.pretraining_tp > 1: #multi nodes
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            #print("hi")
            #print(hidden_states.shape)
            #print(self.policy_head.weight.shape)
            logits = self.policy_head(hidden_states)
            #print("hi")
            #print(logits.shape)
        logits = logits.float()

        loss = None

        check_tensor_values(labels, 'labels')

        #rewards = labels
        #budgets = labels

       
        if labels is not None:
            """
            # Shift so that tokens < n predict n 保证对齐，因为预测值里不应该有最后一个，label里不应该有第一个
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            #loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.action_space_size)
            shift_labels = shift_labels.view(-1)

            #check_tensor_values(shift_labels)
            #check_tensor_values()
            """
            logits_probs = F.softmax(logits, dim=1)
            distribution = Categorical(logits_probs)
            labels = labels.to(logits.device)
            label_masks = label_masks.to(logits.device)
            reward_returns = reward_returns.to(logits.device)
            human_decays = human_decays.to(logits.device)

            

            """
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            rewards = reward_returns[..., 1:].contiguous()
            human_decays = human_decays[..., 1:].contiguous()
            label_masks = label_masks[..., 1:].contiguous()

            check_tensor_values(shift_labels, 'shift_labels')
            #print(shift_logits)
            """
            #check_tensor_values(shift_logits, 'shift_logits')


            #gradient ascend 貌似不用mask
            loss = -distribution.log_prob(labels)*label_masks*(reward_returns-human_decays)
            
            """CRL,lagrange
            loss = -distribution.log_prob(labels)*label_masks*(reward_returns-lamb*(label*label_masks))
            lamb = lamb + alpha*(torch.sum(human_decays)-d)

            """
            loss = torch.sum(loss)
            #loss = loss_fct(shift_logits, shift_labels)

        #print("loss")
        #print(loss.shape)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PolicyOutput(
            loss = loss,
            logits = logits
        )


    def inference(self,input_ids):
        
        outputs = self.model(
            input_ids=input_ids
        )

        
        #torch.save(hidden_states, 'tensor.pth')
        hidden_states = torch.load('tensor.pth')
        hidden_states = hidden_states.to(outputs[0].device)
        #print(hidden_states)
        logits = self.policy_head(hidden_states)
        logits = logits.float()
        print(logits)
        #print(self.policy_head.modules.weight)
        


        last_action_logit = logits[:,-1,:]  # get the <solver> last sub token and predict it [bsz, 1, 2]
        #print(last_action_logit)
        last_action_prob = F.softmax(last_action_logit, dim=1)
        distribution = Categorical(last_action_prob)
        #print(distribution.probs)
        action = distribution.sample()
        return action





