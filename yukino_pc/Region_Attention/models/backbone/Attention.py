#coding: utf-8
#----- 標準ライブラリ -----#
import copy

#----- 専用ライブラリ -----#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter

#----- 自作モジュール -----#
#None

class MultiheadAttention(nn.Module):
    def __init__(self,inputs_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.inputs_dim = inputs_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Linear(inputs_dim, embed_dim)
        self.k_proj = nn.Linear(inputs_dim, embed_dim)
        self.v_proj = nn.Linear(inputs_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        #Attention Map Drop out
        self.Attention_Dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, key_padding_mask=None):
        d_k = q.size()[-1]


        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / d_k ** 0.5

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(key_padding_mask[:, None, None], -9e15)
        attention = F.softmax(attn_logits, dim=-1)

        #Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        values = torch.matmul(attention, v)

        return values, attention

    def forward(self, key, query, value, key_padding_mask=None):
        batch_size, seq_length, embed_dim = value.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Separate Q, K, V from linear output
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]

        # Attention Map calcurate
        values, attention = self.scaled_dot_product(q, k, v, key_padding_mask=key_padding_mask)

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        output = self.output_proj(values)

        return output, attention

class Self_Space_Attention(nn.Module):
    def __init__(self, inputs_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.embed_dim = embed_dim

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Conv2d(inputs_dim, embed_dim * 3, kernel_size=1, padding=0)
        self.output_proj = nn.Conv2d(embed_dim, inputs_dim, kernel_size=1, padding=0)

        #Attention Map Drop out
        self.Attention_Dropout = nn.Dropout(dropout)

        # residual coefficient
        self.alpha = nn.Parameter(torch.zeros([1], dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, channel, H, W = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, self.embed_dim * 3, -1)
        qkv = qkv.transpose(1, 2) #[batch, H*W, dims*3]
        
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # [batch, H*W, dims]

        d_k = q.shape[-1]

        attn_logits = self.inner_product(q, k)
        attn_logits = attn_logits / d_k ** 0.5

        attention = F.softmax(attn_logits, dim=-1)

        #Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        values = torch.bmm(attention, v)  # [batch, H*W, dims]

        values = values.reshape(batch_size, H, W, self.embed_dim)
        values = values.permute(0, 3, 1, 2)  # [batch, dims, H, W]

        values = self.output_proj(values)

        return x + self.alpha * values

    def inner_product(self, q, k):
        """atttention map calculater

        Args:
            q (torch.tensor[batch,H*W,dims]): query
            k (torch.tensor[batch,H*W,dims]): key

        Returns:
            torch.tensor[batch,H*W,H*W]: attention map
        """        
        return torch.bmm(q, k.transpose(-2, -1))

class Self_Space_Region_Attention(nn.Module):
    def __init__(self, inputs_dim, embed_dim, num_heads, num_regions, dropout=0.1):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_regions = num_regions  #split in one head
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Conv2d(inputs_dim, embed_dim * 3, kernel_size=1, padding=0)
        self.output_proj = nn.Conv2d(embed_dim, inputs_dim, kernel_size=1, padding=0)

        self.Rqk_proj = nn.Sequential(
            nn.Conv2d(inputs_dim, num_regions * num_heads * 2, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
        )

        #Attention Map Drop out
        self.Attention_Dropout = nn.Dropout(dropout)

        # residual coefficient
        self.alpha = nn.Parameter(torch.zeros([1], dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.Rqk_proj[0].weight)
        self.Rqk_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, channel, H, W = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, self.num_heads, self.head_dim * 3, -1)
        qkv = qkv.transpose(2, 3) #[batch, Head, H*W, dims*3]
        
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # [batch, Head, H*W, dims]

        Rqk = self.Rqk_proj(x)
        Rqk = Rqk.reshape(batch_size, self.num_heads, self.num_regions * 2, -1)
        Rqk = Rqk.transpose(2, 3)  # [batch, Head, H*W, num_regions*2]

        Rq, Rk = torch.chunk(Rqk, 2, dim=-1)  # [batch, Head, H*W, num_regions]

        q = torch.matmul(q.transpose(-1, -2), Rq)  #[batch, Head, dims, num_q_regions]
        k = torch.matmul(k.transpose(-1, -2), Rk)  #[batch, Head, dims, num_k_regions]
        v = torch.matmul(v.transpose(-1, -2), Rk)  #[batch, Head, dims, num_k_regions]
        
        d_k = q.shape[-2]

        attn_logits = self.inner_product(q, k)  #[batch, Head, num_q_regions, num_k_regions]
        attn_logits = attn_logits / d_k ** 0.5

        attention = F.softmax(attn_logits, dim=-1)

        #Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        values = torch.matmul(v, attention.transpose(-1, -2))  # [batch Head, dims, num_q_regions]
        values = torch.matmul(values, Rq.transpose(-1, -2))  # [batch Head, dims, H*W]

        # [batch, dims, H, W]
        values = values.reshape(batch_size, self.embed_dim, H, W)

        values = self.output_proj(values)

        return x + self.alpha * values

    def inner_product(self, q, k):
        """atttention map calculater

        Args:
            q (torch.tensor[batch, Head, dims, num_q_regions]): query
            k (torch.tensor[batch, Head, dims, num_k_regions]): key

        Returns:
            torch.tensor[batch, Head, num_q_regions, num_k_regions]: attention map
        """
        return torch.matmul(q.transpose(-2, -1), k)


class Src_Tgt_Space_Region_Attention(nn.Module):
    def __init__(self, src_dim, tgt_dim, embed_dim, num_heads, num_regions, dropout=0.1):
        super().__init__()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_regions = num_regions  #split in one head
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = nn.Conv2d(tgt_dim, embed_dim, kernel_size=1, padding=0)
        self.kv_proj = nn.Conv2d(src_dim, embed_dim * 2, kernel_size=1, padding=0)
        self.output_proj = nn.Conv2d(embed_dim, tgt_dim, kernel_size=1, padding=0)

        self.Rq_proj = nn.Sequential(
            nn.Conv2d(tgt_dim, num_regions * num_heads, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
        )
        self.Rk_proj = nn.Sequential(
            nn.Conv2d(src_dim, num_regions * num_heads, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
        )

        #Attention Map Drop out
        self.Attention_Dropout = nn.Dropout(dropout)

        # residual coefficient
        self.alpha = nn.Parameter(torch.zeros([1], dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.Rq_proj[0].weight)
        self.Rq_proj[0].bias.data.fill_(0)

        nn.init.xavier_uniform_(self.Rk_proj[0].weight)
        self.Rk_proj[0].bias.data.fill_(0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def forward(self, src, tgt):
        batch_size, channel, H, W = tgt.size()

        q = self.q_proj(tgt)
        q = q.reshape(batch_size, self.num_heads, self.head_dim, -1)
        q = q.transpose(2, 3)  #[batch, Head, H*W, dims]
        
        kv = self.kv_proj(src)
        kv = kv.reshape(batch_size, self.num_heads, self.head_dim * 2, -1)
        kv = kv.transpose(2, 3)  # [batch, Head, H*W, dims*2]
        
        k, v = torch.chunk(kv, 2, dim=-1)  # [batch, Head, H*W, dims]

        Rq = self.Rq_proj(tgt)
        Rq = Rq.reshape(batch_size, self.num_heads, self.num_regions, -1)
        Rq = Rq.transpose(2, 3)  # [batch, Head, H*W, num_regions]
        
        Rk = self.Rk_proj(src)
        Rk = Rk.reshape(batch_size, self.num_heads, self.num_regions, -1)
        Rk = Rk.transpose(2, 3)  # [batch, Head, H*W, num_regions]

        q = torch.matmul(q.transpose(-1, -2), Rq)  #[batch, Head, dims, num_q_regions]
        k = torch.matmul(k.transpose(-1, -2), Rk)  #[batch, Head, dims, num_k_regions]
        v = torch.matmul(v.transpose(-1, -2), Rk)  #[batch, Head, dims, num_k_regions]
        
        d_k = q.shape[-2]

        attn_logits = self.inner_product(q, k)  #[batch, Head, num_q_regions, num_k_regions]
        attn_logits = attn_logits / d_k ** 0.5

        attention = F.softmax(attn_logits, dim=-1)

        #Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        values = torch.matmul(v, attention.transpose(-1, -2))  # [batch Head, dims, num_q_regions]
        values = torch.matmul(values, Rq.transpose(-1, -2))  # [batch Head, dims, H*W]

        # [batch, dims, H, W]
        values = values.reshape(batch_size, self.embed_dim, H, W)

        values = self.output_proj(values)

        return tgt + self.alpha * values

    def inner_product(self, q, k):
        """atttention map calculater

        Args:
            q (torch.tensor[batch, Head, dims, num_q_regions]): query
            k (torch.tensor[batch, Head, dims, num_k_regions]): key

        Returns:
            torch.tensor[batch, Head, num_q_regions, num_k_regions]: attention map
        """
        return torch.matmul(q.transpose(-2, -1), k)

