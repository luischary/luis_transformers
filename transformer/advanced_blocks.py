import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from flash_attn.flash_attention import FlashAttention

from transformer.transformer_blocks import FeedFowardBlock


class RotaryMultiHeadAttention(nn.Module):
    def __init__(
        self,
        rotary: RotaryEmbedding,
        num_heads: int = 8,
        embed_dim: int = 512,
        dropout=0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert (
            embed_dim % num_heads == 0
        ), "The number of dimensions must be divible by the number of heads"

        self.head_dim = embed_dim // num_heads
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_projection = nn.Dropout(dropout)
        self.rotary = rotary

    def reshape_for_attention(self, x):
        B, L, E = x.shape
        # shape x = [batch, len, embed_dim]
        # precisa virar [batch * heads, len, head_dim]
        x = x.contiguous().view((B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        # virou [batch, heads, len, head_dim]
        # x = x.contiguous().view((B * self.num_heads, L, self.head_dim))
        return x

    def reshape_from_attention(self, x):
        B, H, L, HD = x.shape
        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x = x.transpose(1, 2)
        # virou [batch, len, heads, head_dim]
        x = x.contiguous().view((B, L, self.embed_dim))
        # virou [batch, len, embed_dim]
        return x

    def QKVattention(self, q, k, v, mask=None):
        b, heads, len_tokens, embed_dim = q.shape
        k_t = torch.transpose(k, -1, -2)
        # shapes for q, k, v are [B, HEADS, SEQ, HEAD_DIM]
        # for K_t we have [B, HEADS, HEAD_DIM, SEQ]
        qk = torch.einsum("bhsd, bhde -> bhse", q, k_t)
        # qk = torch.bmm(q, k_t)
        # shape of qk is [B, SEQ, SEQ]
        if mask is not None:
            qk = qk + mask
        attention = torch.softmax(qk / np.sqrt(embed_dim), dim=-1)
        attention = self.dropout_attention(attention)
        # [batch, heads, decoder_len, head_dim] * [batch, heasd, encoder_len, head_dim]
        full_attention = torch.einsum("bhde, bher -> bhdr", attention, v)
        return self.dropout_projection(full_attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        q = self.reshape_for_attention(self.proj_q(x))
        k = self.reshape_for_attention(self.proj_k(x))
        v = self.reshape_for_attention(self.proj_v(x))

        # rotary embeddings
        q = self.rotary.rotate_queries_or_keys(q)
        k = self.rotary.rotate_queries_or_keys(k)

        x_att = self.QKVattention(q, k, v, mask)

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att = self.reshape_from_attention(x_att)

        # projecao final
        x_att = self.out_projection(x_att)
        return x_att


class RotaryMultiHeadFlashAttention(nn.Module):
    def __init__(
        self,
        rotary: RotaryEmbedding,
        num_heads: int = 8,
        embed_dim: int = 512,
        dropout=0.1,
        causal: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.causal = causal

        assert (
            embed_dim % num_heads == 0
        ), "The number of dimensions must be divible by the number of heads"

        self.head_dim = embed_dim // num_heads
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.flash = FlashAttention(attention_dropout=dropout)

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_projection = nn.Dropout(dropout)
        self.rotary = rotary

    def reshape_for_attention(self, x):
        B, L, E = x.shape
        # shape x = [batch, len, embed_dim]
        # virou [batch, heads, len, head_dim]
        x_view = (
            x.contiguous().view((B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        )
        return x_view

    def reshape_from_attention(self, x):
        B, L, H, HD = x.shape
        # virou [batch, len, heads, head_dim]
        x_view = x.contiguous().view(B, L, self.embed_dim)
        # virou [batch, len, embed_dim]
        return x_view

    def forward(self, x: torch.Tensor):
        q = self.reshape_for_attention(self.proj_q(x.clone()))
        k = self.reshape_for_attention(self.proj_k(x.clone()))
        v = self.reshape_for_attention(self.proj_v(x.clone()))

        # rotary embeddings
        if self.rotary.use_xpos:
            q_rotary, k_rotary = self.rotary.rotate_queries_and_keys(q, k, seq_dim=-2)
        else:
            q_rotary = self.rotary.rotate_queries_or_keys(q)
            k_rotary = self.rotary.rotate_queries_or_keys(k)

        # junta qkv
        # precisa que o shape seja [batch, seq, kqv, head, head_dim]
        # mas estamos em [batch, head, seq, head_dim]
        qkv = torch.cat(
            [
                q_rotary.transpose(1, 2).unsqueeze(2),
                k_rotary.transpose(1, 2).unsqueeze(2),
                v.transpose(1, 2).unsqueeze(2),
            ],
            dim=2,
        ).half()  # needed for flash attention
        x_att = self.flash(
            qkv,
            key_padding_mask=None,
            causal=self.causal,
            cu_seqlens=None,
            max_s=None,
            need_weights=None,
        )[0]
        # para inferencia
        if not self.training:
            # se o modelo esta em float32 precisa voltar para float32
            for parametro in self.out_projection.parameters():
                x_att_ajustado = x_att.type(parametro.dtype)
                break
        else:
            x_att_ajustado = x_att

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att_reshaped = self.reshape_from_attention(x_att_ajustado)

        # projecao final
        x_att_projected = self.out_projection(x_att_reshaped)
        return x_att_projected


class RotaryMHFlashCrossAttention(nn.Module):
    def __init__(
        self,
        rotary: RotaryEmbedding,
        num_heads: int = 8,
        embed_dim: int = 512,
        dropout=0.1,
        causal: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.causal = causal

        assert (
            embed_dim % num_heads == 0
        ), "The number of dimensions must be divible by the number of heads"

        self.head_dim = embed_dim // num_heads
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.flash = FlashAttention(attention_dropout=dropout)

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_projection = nn.Dropout(dropout)
        self.rotary = rotary

    def reshape_for_attention(self, x):
        B, L, E = x.shape
        # shape x = [batch, len, embed_dim]
        # virou [batch, heads, len, head_dim]
        x = x.contiguous().view((B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        return x

    def reshape_from_attention(self, x):
        B, L, H, HD = x.shape
        # virou [batch, len, heads, head_dim]
        x = x.contiguous().view(B, L, self.embed_dim)
        # virou [batch, len, embed_dim]
        return x

    def pad_sequence(self, t: torch.Tensor, len: int, padding_value: int = 0):
        B, H, L, D = t.shape
        if L < len:
            len_diff = len - L
            padding = torch.ones((B, H, len_diff, D)) * padding_value
            padded = torch.cat([t, padding.to(t.device)], dim=-2)
            # padding completo
            padding = torch.cat([torch.ones_like(t), padding.to(t.device)], dim=-2)
        else:
            padded = t
            padding = torch.ones_like(t)
        return padded, padding

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor):
        q = self.reshape_for_attention(self.proj_q(x_decoder))
        k = self.reshape_for_attention(self.proj_k(x_encoder))
        v = self.reshape_for_attention(self.proj_v(x_encoder))

        # rotary embeddings
        if self.rotary.use_xpos:
            # se tiverem comprimentos diferentes precisamos fazer um padding
            max_len = max(q.shape[-2], k.shape[-2])
            q_padded, _ = self.pad_sequence(q, max_len)
            k_padded, padding_k = self.pad_sequence(k, max_len)
            v_padded, _ = self.pad_sequence(v, max_len)
            q, k = self.rotary.rotate_queries_and_keys(q_padded, k_padded, seq_dim=-2)
        else:
            q = self.rotary.rotate_queries_or_keys(q)
            k = self.rotary.rotate_queries_or_keys(k)

        qkv = torch.cat(
            [
                q_padded.transpose(1, 2).unsqueeze(2),
                k_padded.transpose(1, 2).unsqueeze(2),
                v_padded.transpose(1, 2).unsqueeze(2),
            ],
            dim=2,
        ).half()

        key_padding_mask = padding_k.transpose(2, 1)[:, :, 0, 0]

        x_att = self.flash(qkv, key_padding_mask=key_padding_mask, causal=False)[0]
        # junta kv
        # precisa que o shape seja [batch, seq, kqv, head, head_dim]
        # mas estamos em [batch, head, seq, head_dim]
        # kv = torch.cat(
        #     [
        #         k.transpose(1, 2).unsqueeze(2),
        #         v.transpose(1, 2).unsqueeze(2),
        #     ],
        #     dim=2,
        # )
        # # arruma o q tambem
        # q = q.transpose(1, 2)

        # batch_size = q.shape[0]
        # q, cu_seqlens_q, max_s_q = self.make_flash_parameters(q)
        # kv, cu_seqlens_kv, max_s_kv = self.make_flash_parameters(kv)

        # x_att = flash_attn_unpadded_kvpacked_func(
        #     q,
        #     kv,
        #     cu_seqlens_q=cu_seqlens_q,
        #     cu_seqlens_k=cu_seqlens_kv,
        #     max_seqlen_q=max_s_q,
        #     max_seqlen_k=max_s_kv,
        #     dropout_p=self.dropout if self.training else 0,
        #     causal=self.causal,
        # )
        # x_att = rearrange(x_att, "(b s) ... -> b s ...", b=batch_size)

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att = self.reshape_from_attention(x_att)

        # projecao final
        x_att = self.out_projection(x_att)
        return x_att

    def make_flash_parameters(self, qkv: torch.Tensor):
        batch_size = qkv.shape[0]
        seqlen = qkv.shape[1]
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = seqlen
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=qkv.device,
        )
        return qkv, cu_seqlens, max_s


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        rotary: RotaryEmbedding,
        num_heads: int = 8,
        embed_dim: int = 512,
        dropout=0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        assert (
            embed_dim % num_heads == 0
        ), "The number of dimensions must be divible by the number of heads"

        self.head_dim = embed_dim // num_heads
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_projection = nn.Dropout(dropout)
        self.rotary = rotary

    def QKVattention(self, q, k, v, mask=None):
        b, heads, len_tokens, embed_dim = q.shape
        k_t = torch.transpose(k, -1, -2)
        # shapes for q, k, v are [B, HEADS, SEQ, HEAD_DIM]
        # for K_t we have [B, HEADS, HEAD_DIM, SEQ]
        qk = torch.einsum("bhsd, bhde -> bhse", q, k_t)
        # qk = torch.bmm(q, k_t)
        # shape of qk is [B, SEQ, SEQ]
        if mask is not None:
            qk = qk + mask
        attention = torch.softmax(qk / np.sqrt(embed_dim), dim=-1)
        attention = self.dropout_attention(attention)
        # [batch, heads, decoder_len, head_dim] * [batch, heasd, encoder_len, head_dim]
        full_attention = torch.einsum("bhde, bher -> bhdr", attention, v)
        return self.dropout_projection(full_attention)

    def reshape_for_attention(self, x):
        B, L, E = x.shape
        # shape x = [batch, len, embed_dim]
        # virou [batch, heads, len, head_dim]
        x = x.contiguous().view((B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        return x

    def reshape_from_attention(self, x):
        B, H, L, HD = x.shape
        # virou [batch, len, heads, head_dim]
        x = x.transpose(1, 2)
        x = x.contiguous().view(B, L, self.embed_dim)
        # virou [batch, len, embed_dim]
        return x

    def pad_sequence(self, t: torch.Tensor, len: int, padding_value: int = 0):
        B, H, L, D = t.shape
        if L < len:
            len_diff = len - L
            padding = torch.ones((B, H, len_diff, D)) * padding_value
            padded = torch.cat([t, padding.to(t.device)], dim=-2)
            # padding completo
            padding = torch.cat([torch.ones_like(t), padding.to(t.device)], dim=-2)
        else:
            padded = t
            padding = torch.ones_like(t)
        return padded, padding

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor):
        q = self.reshape_for_attention(self.proj_q(x_decoder))
        k = self.reshape_for_attention(self.proj_k(x_encoder))
        v = self.reshape_for_attention(self.proj_v(x_encoder))

        # rotary embeddings
        if self.rotary.use_xpos:
            # se tiverem comprimentos diferentes precisamos fazer um padding
            max_len = max(q.shape[-2], k.shape[-2])
            q_padded, _ = self.pad_sequence(q, max_len)
            k_padded, padding_k = self.pad_sequence(k, max_len)
            v_padded, _ = self.pad_sequence(v, max_len)
            q, k = self.rotary.rotate_queries_and_keys(q_padded, k_padded, seq_dim=-2)
        else:
            q = self.rotary.rotate_queries_or_keys(q)
            k = self.rotary.rotate_queries_or_keys(k)

        # atencao
        x_att = self.QKVattention(q=q, k=k, v=v)

        # projecao final
        x_att = self.reshape_from_attention(x_att)
        x_att = self.out_projection(x_att)
        return x_att


class RotaryFlashDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, rotary, dropout: int = 0.1):
        super().__init__()

        self.attention = RotaryMultiHeadFlashAttention(
            rotary=rotary,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            causal=True,
        )
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop_skip_1(x) + self.attention(self.norm_1(x))
        x = self.drop_skip_2(x) + self.feedforward(self.norm_2(x))
        return x


class RotaryFlashCrossDecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        rotary,
        dropout: int = 0.1,
        use_cross_flash: bool = True,
    ):
        super().__init__()

        self.attention = RotaryMultiHeadFlashAttention(
            rotary=rotary,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            causal=True,
        )
        # opcao de usar cross attention com flash ou nao
        if use_cross_flash:
            self.cross_attention = RotaryMHFlashCrossAttention(
                rotary=rotary,
                num_heads=num_heads,
                embed_dim=embed_dim,
                dropout=dropout,
                causal=False,
            )
        else:
            self.cross_attention = RotaryCrossAttention(
                rotary=rotary, num_heads=num_heads, embed_dim=embed_dim, dropout=dropout
            )
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_3 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)
        self.drop_skip_3 = nn.Dropout(dropout)

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor):
        # self attention
        x_decoder = self.drop_skip_1(x_decoder) + self.attention(self.norm_1(x_decoder))
        # cross attention
        x_decoder = self.drop_skip_2(x_decoder) + self.cross_attention(
            self.norm_2(x_decoder), x_encoder
        )
        # feed forward
        x_decoder = self.drop_skip_3(x_decoder) + self.feedforward(
            self.norm_3(x_decoder)
        )
        return x_decoder
