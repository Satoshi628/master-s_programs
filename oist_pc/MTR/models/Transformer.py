#coding: utf-8
#----- Standard Library -----#
import copy

#----- Public Package -----#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter

#----- Module -----#
# None

# linner -> conv1*1
class Transformer_3D(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 move_limit=25.,
                 Encoder="Normal",
                 **kwargs):
        super().__init__()
        
        if Encoder=="Normal":
            encoder_layer = TransformerEncoderLayer_Normal(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                **kwargs)
            decoder_layer = TransformerDecoderLayer_Normal(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                **kwargs)
        elif Encoder == "Distance":
            encoder_layer = TransformerEncoderLayer_Dist(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                move_limit,
                **kwargs)
            decoder_layer = TransformerDecoderLayer_Dist(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                move_limit,
                **kwargs)
        elif Encoder == "Time":
            encoder_layer = TransformerEncoderLayer_Time(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                move_limit,
                **kwargs)
            decoder_layer = TransformerDecoderLayer_Time(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                move_limit,
                **kwargs)
        else:
            raise ValueError("Encoder has values other than Normal, Distance and Time.")

        self.transformerlayer = TransformerLayer(encoder_layer, decoder_layer, num_encoder_layers)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dict_tensor):
        """Transformer

        Args:
            src (dict(torch.tensor)): keys are "encoder", "pos", "mask", "coord"

        dict:
            "encoder" (tensor[batch, T, num, d_model]): inputs vector
            "pos" (tensor[batch, T, num, d_model]): positional encoding vector
            "mask" (tensor[batch, T, num]): PAD mask. True is mask 
            "coord" (tensor[batch, T, num, 2(x, y)]): inputs vector's coordinate

        Returns:
            tensor[batch, T, num, d_model]: output vector
        """
        #transpose [batch, d_model, T, num]
        dict_tensor["encoder"] = dict_tensor["encoder"].permute(0, 3, 1, 2)
        dict_tensor["decoder"] = dict_tensor["decoder"].permute(0, 3, 1, 2)
        dict_tensor["pos"] = dict_tensor["pos"].permute(0, 3, 1, 2)

        output = self.transformerlayer(dict_tensor)
        output = output.permute(0, 2, 3, 1)
        return output



# linner -> conv1*1
class Transformer_3D_encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 move_limit=25.,
                 Encoder="Normal",
                 **kwargs):
        super().__init__()
        
        if Encoder=="Normal":
            encoder_layer = TransformerEncoderLayer_Normal(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                **kwargs)
        elif Encoder == "Distance":
            encoder_layer = TransformerEncoderLayer_Dist(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                move_limit,
                **kwargs)
        elif Encoder == "Time":
            encoder_layer = TransformerEncoderLayer_Time(d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                move_limit,
                **kwargs)
        else:
            raise ValueError("Encoder has values other than Normal, Distance and Time.")

        self.transformerlayer = TransformerEncoder(encoder_layer, num_encoder_layers)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dict_tensor):
        """Transformer

        Args:
            src (dict(torch.tensor)): keys are "encoder", "pos", "mask", "coord"

        dict:
            "encoder" (tensor[batch, T, num, d_model]): inputs vector
            "pos" (tensor[batch, T, num, d_model]): positional encoding vector
            "mask" (tensor[batch, T, num]): PAD mask. True is mask 
            "coord" (tensor[batch, T, num, 2(x, y)]): inputs vector's coordinate

        Returns:
            tensor[batch, T, num, d_model]: output vector
        """
        #transpose [batch, d_model, T, num]
        dict_tensor["encoder"] = dict_tensor["encoder"].permute(0, 3, 1, 2)
        dict_tensor["pos"] = dict_tensor["pos"].permute(0, 3, 1, 2)

        output = self.transformerlayer(dict_tensor)
        output = output.permute(0, 2, 3, 1)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, num_layers):
        super().__init__()
        self.encoder_layers = _get_clones(encoder_layer, num_layers)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.num_layers = num_layers

    def forward(self, dict_tensor):
        for en_layer, de_payer in zip(self.encoder_layers, self.decoder_layers):
            dict_tensor["encoder"] = en_layer(dict_tensor)
            dict_tensor["decoder"] = de_payer(dict_tensor)

        return dict_tensor["decoder"]



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, dict_tensor):
        for layer in self.layers:
            dict_tensor["encoder"] = layer(dict_tensor)
        return dict_tensor["encoder"]


class TransformerEncoderLayer_Normal(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 **kwargs):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout, **kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = Layernorm2d(d_model)
        self.norm2 = Layernorm2d(d_model)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, dict_tensor):
        src2 = self.norm1(dict_tensor["encoder"])
        q = k = self.with_pos_embed(src2, dict_tensor['pos'])

        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]

        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout1(src2)

        src2 = self.norm2(dict_tensor["encoder"])
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout2(src2)
        return dict_tensor["encoder"]


class TransformerEncoderLayer_Dist(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 move_limit=25.,
                 **kwargs):
        super().__init__()
        self.self_attn = Dist_MultiheadAttention(d_model, nhead, dropout, move_limit, **kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = Layernorm2d(d_model)
        self.norm2 = Layernorm2d(d_model)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, dict_tensor):
        src2 = self.norm1(dict_tensor["encoder"])

        q = k = self.with_pos_embed(src2, dict_tensor['pos'])
        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]
        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout1(src2)

        src2 = self.norm2(dict_tensor["encoder"])
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout2(src2)
        return dict_tensor["encoder"]

class TransformerEncoderLayer_Time(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 move_limit=25.,
                 **kwargs):
        super().__init__()
        self.self_attn = Time_MultiheadAttention(d_model, nhead, dropout, move_limit, **kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = Layernorm2d(d_model)
        self.norm2 = Layernorm2d(d_model)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, dict_tensor):
        src2 = self.norm1(dict_tensor["encoder"])

        v = q = k = self.with_pos_embed(src2, dict_tensor['pos'])
        src2 = self.self_attn(q, k, value=v, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]
        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout1(src2)

        src2 = self.norm2(dict_tensor["encoder"])
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout2(src2)
        return dict_tensor["encoder"]


class TransformerEncoderLayer_Both(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 move_limit=25.,
                 **kwargs):
        super().__init__()
        self.self_attn1 = Dist_MultiheadAttention(d_model, nhead, dropout, move_limit, **kwargs)
        self.self_attn2 = Time_MultiheadAttention(d_model, nhead, dropout, move_limit, **kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = Layernorm2d(d_model)
        self.norm2 = Layernorm2d(d_model)
        self.norm3 = Layernorm2d(d_model)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.dropout3 = nn.Dropout2d(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, dict_tensor):
        src2 = self.norm1(dict_tensor["encoder"])

        # distance attention
        q = k = self.with_pos_embed(src2, dict_tensor['pos'])
        src2 = self.self_attn1(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]
        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout1(src2)
        src2 = self.norm2(dict_tensor["encoder"])

        # time attention
        q = k = self.with_pos_embed(src2, dict_tensor['pos'])
        src2 = self.self_attn2(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]
        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout2(src2)
        src2 = self.norm3(dict_tensor["encoder"])
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        dict_tensor["encoder"] = dict_tensor["encoder"] + self.dropout3(src2)
        return dict_tensor["encoder"]


class TransformerDecoderLayer_Normal(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 **kwargs):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout, **kwargs)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout, **kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = Layernorm2d(d_model)
        self.norm2 = Layernorm2d(d_model)
        self.norm3 = Layernorm2d(d_model)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.dropout3 = nn.Dropout2d(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, dict_tensor):
        src2 = self.norm1(dict_tensor["decoder"])
        q = k = self.with_pos_embed(src2, dict_tensor['pos'])

        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]

        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout1(src2)

        src2 = self.norm2(dict_tensor["decoder"])

        q = k = self.with_pos_embed(dict_tensor['encoder'], dict_tensor['pos'])
        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]
        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout1(src2)

        src2 = self.norm3(dict_tensor["decoder"])

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout2(src2)
        return dict_tensor["decoder"]


class TransformerDecoderLayer_Dist(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 move_limit=25.,
                 **kwargs):
        super().__init__()
        self.self_attn = Dist_MultiheadAttention(d_model, nhead, dropout, move_limit, **kwargs)
        self.cross_attn = Dist_MultiheadAttention(d_model, nhead, dropout, move_limit, **kwargs)


        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = Layernorm2d(d_model)
        self.norm2 = Layernorm2d(d_model)
        self.norm3 = Layernorm2d(d_model)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.dropout3 = nn.Dropout2d(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, dict_tensor):
        src2 = self.norm1(dict_tensor["decoder"])
        q = k = self.with_pos_embed(src2, dict_tensor['pos'])

        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]

        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout1(src2)

        src2 = self.norm2(dict_tensor["decoder"])

        q = k = self.with_pos_embed(dict_tensor['encoder'], dict_tensor['pos'])
        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]
        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout1(src2)

        src2 = self.norm3(dict_tensor["decoder"])

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout2(src2)
        return dict_tensor["decoder"]

class TransformerDecoderLayer_Time(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 move_limit=25.,
                 **kwargs):
        super().__init__()
        self.self_attn = Time_MultiheadAttention(d_model, nhead, dropout, move_limit, **kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.linear2 = nn.Conv2d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = Layernorm2d(d_model)
        self.norm2 = Layernorm2d(d_model)
        self.norm3 = Layernorm2d(d_model)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.dropout3 = nn.Dropout2d(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, dict_tensor):
        src2 = self.norm1(dict_tensor["decoder"])
        q = k = self.with_pos_embed(src2, dict_tensor['pos'])

        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]

        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout1(src2)

        src2 = self.norm2(dict_tensor["decoder"])

        q = k = self.with_pos_embed(dict_tensor['encoder'], dict_tensor['pos'])
        src2 = self.self_attn(q, k, value=src2, coord=dict_tensor['coord'], key_padding_mask=dict_tensor['mask'])[0]
        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout1(src2)

        src2 = self.norm3(dict_tensor["decoder"])

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        dict_tensor["decoder"] = dict_tensor["decoder"] + self.dropout2(src2)
        return dict_tensor["decoder"]



class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Conv2d(embed_dim*3, embed_dim*3, kernel_size=1, groups=3)
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Attention Map Drop out
        self.Attention_Dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, key_padding_mask=None):
        d_k = q.size()[-1]

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / d_k ** 0.5

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(key_padding_mask[:, None, :, None], -9e15)
        attention = F.softmax(attn_logits, dim=-1)

        # Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        values = torch.matmul(attention, v)

        return values, attention

    def forward(self, key, query, value, coord, key_padding_mask=None):
        #[batch, d_model, T, num]
        batch_size, embed_dim, time_length, seq_length = value.size()

        qkv = torch.cat([query, key, value], dim=1)
        qkv = self.qkv_proj(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        # Separate Q, K, V from linear output
        q = q.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        q = q.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        k = k.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        k = k.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        v = v.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        v = v.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        # Attention Map calcurate
        values, attention = self.scaled_dot_product(q, k, v, key_padding_mask=key_padding_mask)

        values = values.permute(0, 1, 4, 2, 3)  # [Batch, Head, Dims, T, num]
        values = values.reshape(batch_size, embed_dim, time_length, seq_length)
        output = self.output_proj(values)

        return output, attention


class Dist_MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, move_limit=25., **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.move_limit = move_limit

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Conv2d(embed_dim*3, embed_dim*3, kernel_size=1,groups=3)
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Attention Map Drop out
        self.Attention_Dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, coord, key_padding_mask=None):
        d_k = q.size()[-1]

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        distance = (coord[:, :, :, None] - coord[:, :, None]) ** 2
        distance = torch.sqrt(distance.sum(dim=-1))
        distance = (distance < self.move_limit) * 1.0
        attn_logits = attn_logits / d_k ** 0.5
        
        # Graph attention with short distance
        attn_logits = attn_logits * distance[:, None]

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(key_padding_mask[:, None, :, None], -9e15)
        attention = F.softmax(attn_logits, dim=-1)

        # Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        values = torch.matmul(attention, v)

        return values, attention

    def forward(self, key, query, value, coord, key_padding_mask=None):
        #[batch, d_model, T, num]
        batch_size, embed_dim, time_length, seq_length = value.size()

        qkv = torch.cat([query, key, value], dim=1)
        qkv = self.qkv_proj(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        # Separate Q, K, V from linear output
        q = q.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        q = q.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        k = k.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        k = k.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        v = v.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        v = v.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        # Attention Map calcurate
        values, attention = self.scaled_dot_product(
            q, k, v, coord, key_padding_mask=key_padding_mask)

        values = values.permute(0, 1, 4, 2, 3)  # [Batch, Head, Dims, T, num]
        values = values.reshape(batch_size, embed_dim, time_length, seq_length)
        output = self.output_proj(values)

        return output, attention


class Time_MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, move_limit=25., **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.move_limit = move_limit

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Conv2d(embed_dim*3, embed_dim*3, kernel_size=1,groups=3)
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Attention Map Drop out
        self.Attention_Dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, coord, key_padding_mask=None):
        # [Batch, Head, T, num, Dims]
        # currenct time =0,1,2,...,T-1,T  ##target
        C_q = q
        C_coord = coord
        # next time = 1,2,3,...,T-1,T,T  ##source
        N_k = torch.cat([k[:, :, 1:], k[:, :, -1:]], dim=2)
        N_v = torch.cat([v[:, :, 1:], v[:, :, -1:]], dim=2)
        N_coord = torch.cat([coord[:, :, 1:], coord[:, :, -1:]], dim=2)
        if key_padding_mask is not None:
            N_mask = torch.cat([key_padding_mask[:, :, 1:], key_padding_mask[:, :, -1:]], dim=2)
        # previous time = 0,0,1,2,3,...,T-1  ##source
        P_k = torch.cat([k[:, :, :1], k[:, :, :-1]], dim=2)
        P_v = torch.cat([v[:, :, :1], v[:, :, :-1]], dim=2)
        P_coord = torch.cat([coord[:, :, :1], coord[:, :, :-1]], dim=2)
        if key_padding_mask is not None:
            P_mask = torch.cat([key_padding_mask[:, :, :1], key_padding_mask[:, :, :-1]], dim=2)
        
        d_k = q.size()[-1]

        # forward time attention
        attn_logits = torch.matmul(C_q, P_k.transpose(-2, -1))
        distance = (C_coord[:, :, :, None] - P_coord[:, :, None]) ** 2
        distance = torch.sqrt(distance.sum(dim=-1))
        distance = (distance < self.move_limit) * 1.0
        attn_logits = attn_logits / d_k ** 0.5
        
        # Graph attention with short distance
        attn_logits = attn_logits * distance[:, None]

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(P_mask[:, None, :, None], -9e15)
        attention = F.softmax(attn_logits, dim=-1)

        # Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        forward_values = torch.matmul(attention, P_v)

        # backward time attention
        attn_logits = torch.matmul(C_q, N_k.transpose(-2, -1))
        distance = (C_coord[:, :, :, None] - N_coord[:, :, None]) ** 2
        distance = torch.sqrt(distance.sum(dim=-1))
        distance = (distance < self.move_limit) * 1.0
        attn_logits = attn_logits / d_k ** 0.5
        
        # Graph attention with short distance
        attn_logits = attn_logits * distance[:, None]

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(N_mask[:, None, :, None], -9e15)
        attention = F.softmax(attn_logits, dim=-1)

        # Attention Map Drop out
        attention = self.Attention_Dropout(attention)

        backward_values = torch.matmul(attention, N_v)

        values = (forward_values + backward_values) * 0.5
        return values, attention

    def forward(self, key, query, value, coord, key_padding_mask=None):
        #[batch, d_model, T, num]
        batch_size, embed_dim, time_length, seq_length = value.size()

        qkv = torch.cat([query, key, value], dim=1)
        qkv = self.qkv_proj(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        # Separate Q, K, V from linear output
        q = q.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        q = q.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        k = k.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        k = k.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        v = v.reshape(batch_size, self.num_heads, self.head_dim, time_length, seq_length)
        v = v.permute(0, 1, 3, 4, 2)  # [Batch, Head, T, num, Dims]

        # Attention Map calcurate
        values, attention = self.scaled_dot_product(
            q, k, v, coord, key_padding_mask=key_padding_mask)

        values = values.permute(0, 1, 4, 2, 3)  # [Batch, Head, Dims, T, num]
        values = values.reshape(batch_size, embed_dim, time_length, seq_length)
        output = self.output_proj(values)

        return output, attention


class Layernorm2d(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        self.norm2d = nn.LayerNorm(normalized_shape)

    def forward(self, inputs):
        # inputs.size() => [batch, d_model, T, num]
        output = inputs.permute(0, 2, 3, 1)
        output = self.norm2d(output)
        output = output.permute(0, 3, 1, 2)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

