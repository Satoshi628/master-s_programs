#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
#None

class ResidualBlock_Small(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualBlock_Large(nn.Module):
    expansion = 4
    def __init__(self, in_channel, middle_channel, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, middle_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, middle_channel * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(middle_channel * 4)
        )
        if in_channel == middle_channel * 4:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channel, middle_channel * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(middle_channel * 4)
            )

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_c=3):
        super().__init__()
        channel = 64

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        layers = []
        layers.append(self._make_layer(block, channel, channel, num_blocks[0], stride=1))
        in_c = channel * block.expansion
        channel *= 2
        for num in num_blocks[1:]:
            layers.append(self._make_layer(block, in_c, channel, num, stride=2))
            in_c = channel * block.expansion
            channel *= 2
        
        self.layer = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, in_c, out_c, num_blocks, stride):
        layers = []
        layers.append(block(in_c, out_c, stride))
        in_c = out_c * block.expansion

        last_idx = num_blocks

        for i in range(1, last_idx):
            layers.append(block(in_c, out_c, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer(out)
        return out

    def seg_process(self, x):
        out = self.conv(x)
        out2 = self.layer[0](out)
        out = self.layer[1](out2)
        out = self.layer[2](out)
        out = self.layer[3](out)

        return out, out2
