from typing import Dict, List

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from transformer.hierarchical.encoder import RotaryHierarchicalEncoder
from transformer.advanced_blocks import RotaryFlashCrossDecoderBlock
from transformer.utils import count_parameters
from transformer.text_generation import LMPipeline


class GPTCross(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.rotary = RotaryEmbedding(embed_dim // num_heads, use_xpos=True)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_blocks.append(
                RotaryFlashCrossDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    rotary=self.rotary,
                )
            )

        self.last_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x_decoder, x_encoder):
        x_decoder = self.embedding(x_decoder)

        for block in self.decoder_blocks:
            x_decoder = block(x_decoder, x_encoder)

        return self.last_norm(x_decoder)


class HBART(nn.Module):
    def __init__(self, encoder_params: Dict, decoder_params: Dict):
        super().__init__()

        self.encoder = RotaryHierarchicalEncoder(**encoder_params)
        self.decoder = GPTCross(**decoder_params)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        # pega so os tokens globais do encoder hierarquico
        _, x_encoder = self.encoder(x)
        if x_encoder.dtype not in [torch.float16, torch.bfloat16]:
            x_encoder = x_encoder.type(torch.float16)
        x = self.decoder(x, x_encoder)
        return x


class HBARTLM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, encoder_params: Dict, decoder_params: Dict
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.model = HBART(encoder_params=encoder_params, decoder_params=decoder_params)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, non_pad_indexes=None):
        # we dont want to compute loss for padding tokens
        # get all hidden states
        logits = self.lm_head(self.model(x))
        # remove batch dimension
        logits = torch.reshape(logits, (-1, self.vocab_size))
        # get only the tokens that matter
        if non_pad_indexes is not None:
            logits = logits[non_pad_indexes, :]

        return logits
