import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
from transformers import AutoConfig, AutoModel
from transformers.models.esm.configuration_esm import EsmConfig


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model.model, output_hidden_states=True
            )
            self.config.hidden_dropout = 0.0
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_dropout = 0.0
            self.config.attention_probs_dropout_prob = 0.0
            # logger.info(self.config)
        else:
            add_safe_globals([EsmConfig])
            self.config = torch.load(config_path, weights_only=False)
            # self.config = AutoConfig.from_pretrained(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        if self.cfg.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc1 = nn.Linear(self.config.hidden_size, self.cfg.model.pca_dim)
        self.fc2 = nn.Linear(self.cfg.model.pca_dim * 6, 1)
        self._init_weights(self.fc1)
        self._init_weights(self.fc2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs, position):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, position)
        return feature

    def forward(self, inputs1, inputs2, position):
        feature1 = self.fc1(self.feature(inputs1, position))
        feature2 = self.fc1(self.feature(inputs2, position))

        feature3 = self.fc1(self.feature(inputs1, inputs1["attention_mask"]))
        feature4 = self.fc1(self.feature(inputs2, inputs2["attention_mask"]))

        feature = torch.cat(
            (
                feature1,
                feature2,
                feature2 - feature1,
                feature3,
                feature4,
                feature4 - feature3,
            ),
            axis=-1,
        )

        output = self.fc2(feature)
        return output
