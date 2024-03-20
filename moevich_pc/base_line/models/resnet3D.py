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
        if stride == 1:
            pad = (0, 1, 1)
            sc_pad = 0
        if stride == 2:
            pad = (2, 1, 1)
            sc_pad = (2, 0, 0)
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(1, 3, 3), stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=(1, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, padding=sc_pad, bias=False),
                nn.BatchNorm3d(out_channel)
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
        if stride ==1:
            pad = (0, 1, 1)
            sc_pad = 0
        if stride == 2:
            pad = (2, 1, 1)
            sc_pad = (2, 0, 0)

        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Conv3d(out_channel, out_channel, kernel_size=(1, 3, 3), stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Conv3d(out_channel, out_channel * 4, kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channel * 4),
            nn.ReLU(inplace=True)
        )
        
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel * 4, kernel_size=1, stride=stride, padding=sc_pad, bias=False),
            nn.BatchNorm3d(out_channel * 4)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    """C2D with ResNet 50 backbone.
    The only operation involving the temporal domain are the pooling layer after the second residual block.
    For more details of the structure, refer to Table 1 from the paper. 
    Padding was added accordingly to match the correct dimensionality.
    """
    def __init__(self, block, num_blocks, non_local, in_c=3, num_classes=1):
        super().__init__()
        channel = 64

        # first convolution operation has essentially 2D kernels
        # output: 64 x 16 x 112 x 112
        self.conv = nn.Sequential(
            nn.Conv3d(3, channel, kernel_size=(1, 7, 7), stride=2, padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1)
        )
        layers = []

        # output: 64 x 8 x 56 x 56
        layers.append(self._make_layer(block, channel, channel, num_blocks[0], stride=1, non_local=non_local[0]))
        in_c = channel * block.expansion
        channel *= 2

        # output: 256 x 4 x 56 x 56
        layers.append(nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)))

        for num, flag in zip(num_blocks[1:], non_local[1:]):
            layers.append(self._make_layer(block, in_c, channel, num, stride=2, non_local=flag))
            in_c = channel * block.expansion
            channel *= 2
        
        self.layer = nn.Sequential(*layers)
        # output: 2048 x 1
        self.linear = nn.Linear(in_c, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, in_c, out_c, num_blocks, stride=1, non_local=False):
        layers = []
        layers.append(block(in_c, out_c, stride))
        in_c = out_c * block.expansion
        
        last_idx = num_blocks
        if non_local:
            last_idx = num_blocks - 1
            
        for i in range(1, last_idx):
            layers.append(block(in_c, out_c))
        
        # add non-local block here
        if non_local:
            layers.append(NLBlockND(in_channels=in_c, dimension=3))
            layers.append(block(in_c, out_c))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer(x)

        x = x.mean(dim=(-3, -2, -1))
        x = self.linear(x)

        return x


def resnet3D50(in_c=3, num_classes=1, non_local=False):

    model = ResNet3D(ResidualBlock_Large,
                    [3, 4, 6, 3],
                    non_local=[False, False, non_local, False],
                    in_c=in_c,
                    num_classes=num_classes)
    return model


def resnet3D34(in_c=3, num_classes=1, non_local=False):
    model = ResNet3D(ResidualBlock_Small, [3, 4, 6, 3],
                    non_local=[False, False, non_local, False],
                    in_c=in_c,
                    num_classes=num_classes)
    
    return model


def resnet3D18(in_c=3, num_classes=1, non_local=False):
    model =  ResNet3D(ResidualBlock_Small, [2, 2, 2, 2],
                    non_local=[False, False, non_local, False],
                    in_c=in_c,
                    num_classes=num_classes)
    return model


if __name__=='__main__':
    # Test case of 32 frames (224 x 224 x 3) input of batch size 1
    img = torch.randn(1, 3, 32, 224, 224)
    net = resnet3D50(non_local=True)
    count = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count += 1
            print(name)
    print (count)
    out = net(img)
    print(out.size())
