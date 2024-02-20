import dataclasses
from typing import Dict
from transformers.trainer import *
from transformers import Trainer, AutoModelForCausalLM
from transformers.utils import is_peft_available
# from trl.trainer.utils import PeftSavingCallback

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

class PolicyTrainer(Trainer):
    r"""
    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`transformers.TrainingArguments`]):
            The arguments to tweak for training. Please refer to the official documentation of `transformers.TrainingArguments`
            for more information.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.        
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[DataCollator] = None,
        peft_config: Optional[Dict] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                if not isinstance(model, PreTrainedModel):
                    model = AutoModelForCausalLM.from_pretrained(
                        model,
                    )

                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                    model = prepare_model_for_kbit_training(
                        model, use_gradient_checkpointing=args.gradient_checkpointing
                    )

                    args = dataclasses.replace(args, gradient_checkpointing=False)

                model = get_peft_model(model, peft_config)  
            model.print_trainable_parameters()


        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss