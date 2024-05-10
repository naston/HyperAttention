import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from .model_factory import Transformer


class LlamaConfig(PretrainedConfig):
    model_type = 'LlamaModel'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

    def update_config(self, config):
        pass


class LlamaModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = Transformer(config)

    def forward(self, input_ids, labels):
        logits = self.model(input_ids)
        train_loss = self.loss_func(logits.transpose(1,2), labels)

        return CausalLMOutput(train_loss, logits=logits)