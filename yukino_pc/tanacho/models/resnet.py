#coding: utf-8
#----- 標準ライブラリ -----#
#None

#----- 専用ライブラリ -----#
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#----- 自作モジュール -----#
#None

class ResNet(nn.Module):
    def __init__(self, resnet_type="resnet18", pretrained=True):
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            self.resnet_model = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet_model = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            self.resnet_model = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            self.resnet_model = torchvision.models.resnet152(pretrained=pretrained)

        # latest channel
        self.feature_layer = nn.Linear(self.resnet_model.layer4[-1].conv2.weight.shape[1], 512)

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)  # width, heightは1/4
        x = self.resnet_model.layer2(x)  # width, heightは1/8
        x = self.resnet_model.layer3(x)  # width, heightは1/16
        out = self.resnet_model.layer4(x)  # width, heightは1/32
        out = out.mean(dim=(-1, -2))

        feature = self.feature_layer(out)
        return feature



class ResNet_avgpool4(nn.Module):
    def __init__(self, resnet_type="resnet18", pretrained=True):
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            self.resnet_model = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet_model = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            self.resnet_model = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            self.resnet_model = torchvision.models.resnet152(pretrained=pretrained)

        # latest channel
        self.avgpool = nn.AvgPool2d(4)
        self.feature_layer = nn.Linear(self.resnet_model.layer4[-1].conv2.weight.shape[1] * 4 * 4, 512)

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)  # width, heightは1/4
        x = self.resnet_model.layer2(x)  # width, heightは1/8
        x = self.resnet_model.layer3(x)  # width, heightは1/16
        out = self.resnet_model.layer4(x)  # width, heightは1/32

        out = self.avgpool(out)
        out = out.flatten(-3)

        feature = self.feature_layer(out)
        return feature

