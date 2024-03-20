#coding: utf-8
##### ライブラリ読み込み #####
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as ttf
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

############################


class P2P_CrossEntoropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, use_Focal=True):
        outputs = torch.sigmoid(outputs)
        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
        #print("targets:",targets.size())
        #print("outputs:", outputs.size())
        #input()
        
        loss = targets * torch.log(outputs) + \
            (1 - targets) * torch.log(1 - outputs)
        # print("loss",loss.size())
        # input()
        # weightはどれだけ本物に近づいたかを表す。値が低いほど本物
        weight = float(use_Focal) * torch.abs(outputs -
                                              targets) + 1 - float(use_Focal)
        loss = loss * weight
        #print(loss.mean().shape)
        #input()
        return - loss.mean()



class P2P_Image():
    def __init__(self):
        pass

    def trans(self, inputs):

        inputs = torch.sigmoid(inputs)*255
        inputs = inputs.squeeze()
        inputs = inputs.to('cpu').detach().numpy().astype(np.uint8)
        return inputs


    def save(self, inputs, folder='result', name=''):
        # データの画像化
        inputs = self.trans(inputs)

        fig = plt.figure()
        plt.subplots_adjust(hspace=0)
        for i in range(inputs.shape[0]):
            title = fig.add_subplot(inputs.shape[0], 5, i+1)
            plt.tick_params(labelbottom=False,
                            labelleft=False,
                            labelright=False,
                            labeltop=False,
                            bottom=False,
                            left=False,
                            right=False,
                            top=False)
            plt.imshow(inputs[i], cmap='magma')

        plt.savefig(
            '{}//images/Confidence_image/Confidence_epoch_{:d}.png'.format(folder, name))
        return
