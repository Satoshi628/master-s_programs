#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
from models.non_local import NLBlockND


class ResidualBlock_Small(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
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
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * 4)
        )
        
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * 4)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, block, num_blocks, non_local, in_c=3, num_classes=1):
        super().__init__()
        channel = 64

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        layers = []
        layers.append(self._make_layer(block, channel, channel, num_blocks[0], stride=1, non_local=non_local[0]))
        in_c = channel * block.expansion
        channel *= 2
        for num, flag in zip(num_blocks[1:], non_local[1:]):
            layers.append(self._make_layer(block, in_c, channel, num, stride=2, non_local=flag))
            in_c = channel * block.expansion
            channel *= 2
        
        self.layer = nn.Sequential(*layers)

        self.linear = nn.Linear(in_c, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, in_c, out_c, num_blocks, stride, non_local=False):
        layers = []
        layers.append(block(in_c, out_c, stride))
        in_c = out_c * block.expansion

        last_idx = num_blocks
        if non_local:
            last_idx = num_blocks - 1

        for i in range(1, last_idx):
            layers.append(block(in_c, out_c, 1))

        if non_local:
            layers.append(NLBlockND(in_channels=in_c, dimension=2))
            layers.append(block(in_c, out_c, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer(out)
        out = out.mean(dim=(-2, -1))
        out = self.linear(out)
        return out


def resnet50(in_c=3, num_classes=1, non_local=False):
    model = ResNet2D(ResidualBlock_Large, [3, 4, 6, 3],
                    non_local=[False, False, non_local, False],
                    in_c=in_c,
                    num_classes=num_classes)
    return model


def resnet34(in_c=3, num_classes=1, non_local=False):
    model =  ResNet2D(ResidualBlock_Small, [3, 4, 6, 3],
                    non_local=[False, False, non_local, False],
                    in_c=in_c,
                    num_classes=num_classes)
    return model


def resnet18(in_c=3, num_classes=1, non_local=False):
    model =  ResNet2D(ResidualBlock_Small, [2, 2, 2, 2],
                    non_local=[False, False, non_local, False],
                    in_c=in_c,
                    num_classes=num_classes)
    return model

"""
if __name__=='__main__':
    # Test case for (224 x 224 x 3) input of batch size 1
    img = torch.randn(1, 3, 224, 224)

    net = resnet2D50()
    count = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count += 1
            print(name)
    print (count)
    out = net(img)
    print(out.size())
"""