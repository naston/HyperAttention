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
        self.decoder = None

    def forward(self, input_ids, labels):
        x = self.model(input_ids)
        decoded = self.softmax(self.decoder(x))
        train_loss = self.loss_func(decoded.transpose(1,2), labels)

        return CausalLMOutput(train_loss, logits=decoded)