#coding: utf-8
#----- 標準ライブラリ -----#
import sys

#----- 専用ライブラリ -----#
import torch
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
#None

##### Conv2d #####
def Conv2d(in_channels=None, out_channels=None, kernel_size=None):
    error_flag = False
    
    if in_channels is None:
        print("Conv2dにはin_channelが必要です")
        print("引数にin_channelを入れてください")
        error_flag = True

    if out_channels is None:
        print("Conv2dにはout_channelsが必要です")
        print("引数にout_channelsを入れてください")
        error_flag = True
    
    if kernel_size is None:
        print("Conv2dにはkernel_sizeが必要です")
        print("引数にkernel_sizeを入れてください")
        error_flag = True

    if error_flag:
        sys.exit(0)

    if kernel_size % 2 == 0:
        print("kernel_sizeは奇数にしてください")
        sys.exit(0)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)


##### MaxPool2d #####
def MaxPool2d():
    return nn.MaxPool2d(2)


##### AvgPool2d #####
def AvgPool2d():
    return nn.AvgPool2d(2)


##### ConvPool2d #####
def ConvPool2d(in_channels=None, out_channels=None):
    error_flag = False

    if in_channels is None:
        print("ConvPool2dにはin_channelが必要です")
        print("引数にin_channelを入れてください")
        error_flag = True

    
    if out_channels is None:
        print("ConvPool2dにはout_channelsが必要です")
        print("引数にout_channelsを入れてください")
        error_flag = True

    if error_flag:
        sys.exit(0)
    
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

##### Mish #####
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))

##### Activations #####
def Activations(function=None):
    if function is None:
        print("Activationsにはfunctionが必要です")
        print("引数にfunctionを入れてください")
        print("例:")
        print("Activations(\"ReLU\")")
        sys.exit(0)

    func_dict = {"ELU": nn.ELU(inplace=True),
                    "Mish": Mish(),
                    "ReLU": nn.ReLU(inplace=True),
                    "ReLU6": nn.ReLU6(inplace=True),
                    "Leaky ReLU": nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    "GELU": nn.GELU(),
                    "Sigmoid": nn.Sigmoid(),
                    "Tanh": nn.Tanh()}

    if func_dict.get(function) is None:
        print(f"function:{function}は使用できません")

        for key in func_dict.keys():
            print(f"\"{key}\"", end=',')
        print("の中から選んでください")
        sys.exit(0)
    
    return func_dict[function]
