#coding: utf-8
#----- Standard Library -----#
import collections

#----- Public Package -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

#----- Module -----#
#None

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple([x for _ in range(n)])
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Sine_Positional_Encoding(nn.Module):
    def __init__(self, d_model, temperature=10000, **kwargs):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number.")
        self.d_model = d_model // 2
        self.temperature = temperature

    def forward(self, coordinate, size):
        """ Produce Sine Positional Encoding from coordinate.
        Reference:https://qiita.com/halhorn/items/c91497522be27bde17ce#positional-encoding

        Args:
            coordinate (tensor[batch,T,num,2(x,y)], long): detection coordinate
            size (list[2(H, W)]): input image size

        Returns:
            tensor[batch,num,d_model]: Sine Positional Encoding
        """        
        
        #dim_i =[0,0,1,1,2,2,3,3,...]
        dim_i = torch.arange(self.d_model, dtype=torch.float32, device=coordinate.device) // 2
        dim_i = 2 * dim_i / self.d_model
        dim_i = self.temperature ** (dim_i)
        
        coord = coordinate.float()

        #pos_x =>size([batch,T,num,d_model/2])
        pos_x = coord[:, :, :, 0, None] * dim_i
        pos_y = coord[:, :, :, 1, None] * dim_i

        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=-1).flatten(-2)
        
        #pos =>size([batch,T,num,d_model])
        pos = torch.cat([pos_x, pos_y], dim=-1)

        pos[coordinate[:, :, :, 0] < 0, :] = 0.

        return pos


class Learned_Positional_Encoding(nn.Module):
    def __init__(self, d_model, resolution=256, **kwargs):
        """ Learnable Positional Encoding

        Args:
            d_model (int): vector dim
            size (int or list or tuple): input image size[x,y]
            resolution (int or list or tuple): input image resolution[x,y]

        Raises:
            ValueError: d_model is divisible by 2
        """        
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number.")
        self.d_model = d_model // 2
        self.resolution = _pair(resolution)

        self.x_embedding = nn.Parameter(torch.empty(self.resolution[0], self.d_model))
        self.y_embedding = nn.Parameter(torch.empty(self.resolution[1], self.d_model))
        
        xavier_uniform_(self.x_embedding)
        xavier_uniform_(self.y_embedding)

    def forward(self, coordinate, size):
        """ Learnable Positional Encoding

        Args:
            coordinate (tensor[batch,T,num,2(x,y)], long): detection coordinate
            size (list[2(H, W)]): input image size

        Returns:
            tensor[batch,num,d_model]: Learnable Positional Encoding
        """        
        coord_x = self.resolution[0] * coordinate[:, :, :, 0].float() / size[1]
        coord_y = self.resolution[1] * coordinate[:, :, :, 1].float() / size[0]
        coord_x = coord_x.long()
        coord_y = coord_y.long()

        pos = torch.cat([self.x_embedding[coord_x], self.y_embedding[coord_y]], dim=-1)

        #maskのかかった部分は0にする
        pos[coordinate[:, :, :, 0] < 0, :] = 0.

        return pos


class Linear_Positional_Encoding(nn.Module):
    def __init__(self, d_model, input_channel=2, dim_feedforward_layers=[32, 64, 128, 256], **kwargs):
        super().__init__()
        self.encoder = MLP([input_channel, *dim_feedforward_layers, d_model])

    def forward(self, coordinate, size):
        """ Create linear Positional Encoding from coordinate
        Reference:https://github.com/magicleap/SuperGluePretrainedNetwork/blob/c0626d58c843ee0464b0fa1dd4de4059bfae0ab4/models/superglue.py#L73

        Args:
            coordinate (tensor[batch,T,num,2(x,y)],long): detection coordinate
            size (list[2(H, W)]): input image size
        Returns:
            tensor[batch,T,num,d_model]: linear Positional Encoding
        """        
        inputs = coordinate.permute(0, 3, 1, 2).float()
        
        # normalized [0~1]
        inputs[:, 0, :, :] = inputs[:, 0, :, :] / size[1]
        inputs[:, 1, :, :] = inputs[:, 1, :, :] / size[0]
        pos = self.encoder(inputs)
        pos = pos.permute(0, 2, 3, 1)

        pos[coordinate[:, :, :, 0] < 0, :] = 0.

        return pos


def MLP(channels):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        if i < (n - 1):
            layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=True))
    return nn.Sequential(*layers)
