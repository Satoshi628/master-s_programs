


_BACKBONES = {
    "alexnet": "models.alexnet(weights=models.alexnet_Weights.DEFAULT)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(weights=models.resnet50_Weights.DEFAULT)",
    "resnet101": "models.resnet101(weights=models.resnet101_Weights.DEFAULT)",
    "resnext101": "models.resnext101_32x8d(weights=models.resnext101_Weights.DEFAULT)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(weights=models.vgg11_Weights.DEFAULT)",
    "vgg19": "models.vgg19(weights=models.vgg19_Weights.DEFAULT)",
    "vgg19_bn": "models.vgg19_bn(weights=models.vgg19_bn_Weights.DEFAULT)",
    "wideresnet50": "models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)",
    "wideresnet101": "models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.DEFAULT)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}

class LastLayerToExtractReachedException(Exception):
    pass
