import math
import torch
from torch import nn
from networks.util import ResBlock


class EncoderVqResnet32(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet32, self).__init__()
        self.flg_variance = flg_var_q
        # Convolution layers
        layers_conv = []
        layers_conv.append(nn.Conv2d(3, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU(True))
        self.conv = nn.Sequential(*layers_conv)
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVqResnet32(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet32, self).__init__()
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        # Convolution layers
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)

        return out


class DecoderResnet32_multi(nn.Module):
    def __init__(self, scales, channels, cfgs, flg_bn=True):
        super(DecoderResnet32_multi, self).__init__()
        # Resblocks
        num_rb = cfgs.num_rb
        res_list = nn.ModuleList()
        past_c_dim = None
        for c_dim in channels:
            layers_resblocks = []
            if past_c_dim is not None:
                layers_resblocks.append(nn.ConvTranspose2d(c_dim + past_c_dim, c_dim, 3, stride=1, padding=1))
                if flg_bn:
                    layers_resblocks.append(nn.BatchNorm2d(c_dim))
            
            for i in range(num_rb):
                layers_resblocks.append(ResBlock(c_dim))
            
            layers_resblocks.append(nn.ConvTranspose2d(c_dim, c_dim, 3, stride=1, padding=1))
            if flg_bn:
                layers_resblocks.append(nn.BatchNorm2d(c_dim))
            layers_resblocks.append(nn.ReLU(True))
            layers_resblocks.append(nn.ConvTranspose2d(c_dim, c_dim // 2, 4, stride=2, padding=1))

            res_list.append(nn.Sequential(*layers_resblocks))
            past_c_dim = c_dim // 2
        
        self.res_list = res_list
        # Convolution layers
        
        scale = int(math.log2(min(scales) // 2))

        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(past_c_dim, past_c_dim, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(past_c_dim))
        layers_convt.append(nn.ReLU(True))
        
        for _ in range(scale-1):
            layers_convt.append(nn.ConvTranspose2d(past_c_dim, past_c_dim // 2, 4, stride=2, padding=1))
            if flg_bn:
                layers_convt.append(nn.BatchNorm2d(past_c_dim // 2))
            layers_convt.append(nn.ReLU(True))
            past_c_dim = past_c_dim // 2

        layers_convt.append(nn.ConvTranspose2d(past_c_dim, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, inputs):
        x = None
        for feature, res in zip(inputs, self.res_list):
            if x is not None:
                feature = torch.cat([feature, x], dim=1)
            x = res(feature)
        out = self.convt(x)

        return out
