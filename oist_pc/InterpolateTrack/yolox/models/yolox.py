#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, heads=None, interpolate=2):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN(interpolate=interpolate)
        if heads is None:
            heads = [YOLOXHead(80) for _ in range(interpolate+1)]
            
        self.backbone = backbone
        self.heads = heads

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        print(self.heads[0](fpn_outs))
        input()
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = 0,0,0,0,0,0

        if self.training:
            assert targets is not None
            for head in self.heads:
                loss_temp, iou_loss_temp, conf_loss_temp, cls_loss_temp, l1_loss_temp, num_fg_temp = head(
                    fpn_outs, targets, x
                )
                loss = loss + loss_temp
                iou_loss = iou_loss + iou_loss_temp
                conf_loss = conf_loss + conf_loss_temp
                cls_loss = cls_loss + cls_loss_temp
                l1_loss = l1_loss + l1_loss_temp
                num_fg = num_fg + num_fg_temp

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            self.heads[0](fpn_outs)
            [head(fpn_outs)for head in self.heads]


        return outputs
