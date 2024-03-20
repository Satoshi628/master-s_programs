import timm  # noqa
import torch
import torchvision.models as models  # noqa
from fastflow import build_FastFlow
from vq_fastflow import build_VQ_FastFlow
def load_ref_wrn50():
    
    import resnet 
    return resnet.wide_resnet50_2(True)

_BACKBONES = {
    "cait_s24_224" : "cait.cait_S24_224(True)",
    "cait_xs24": "cait.cait_XS24(True)",
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet18": "models.resnet18(pretrained=True)",
    "resnet50": "models.resnet50(pretrained=True)",
    "mc3_resnet50": "load_mc3_rn50()", 
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
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
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "ref_wideresnet50": "load_ref_wrn50()",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
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
    "FastFlow_data_new": 'build_FastFlow(pretrain_path="/mnt/kamiya/code/FastFlow/_fastflow_experiment_checkpoints/exp_dn_data_new/109.pt",backbone_name="wide_resnet50_2",flow_steps=8,input_size=[384, 576])',
    "FastFlow_data_new_2": 'build_FastFlow(pretrain_path="/mnt/kamiya/code/FastFlow/_fastflow_experiment_checkpoints/exp_dn_data_new_2/499.pt",backbone_name="wide_resnet50_2",flow_steps=8,input_size=[384, 576])',
    "FastFlow_data_new_3": 'build_FastFlow(pretrain_path="/mnt/kamiya/code/FastFlow/_fastflow_experiment_checkpoints/exp_dn_data_new_3/239.pt",backbone_name="wide_resnet50_2",flow_steps=8,input_size=[384, 576])',
    "FastFlow_data3": 'build_FastFlow(pretrain_path="/mnt/kamiya/code/FastFlow/_fastflow_experiment_checkpoints/exp_dn_data3/489.pt", backbone_name="wide_resnet50_2",flow_steps=8,input_size=[320, 480])',
    "FastFlow_data4": 'build_FastFlow(pretrain_path="/mnt/kamiya/code/FastFlow/_fastflow_experiment_checkpoints/exp_dn_data4/489.pt", backbone_name="wide_resnet50_2",flow_steps=8,input_size=[320, 480])',
    "VQ_FastFlow_data_new": 'build_VQ_FastFlow(pretrain_path="/mnt/kamiya/code/VQ-FastFlow/_fastflow_experiment_checkpoints/exp_dn_data_new_K512/best.pt",backbone_name="wide_resnet50_2",flow_steps=8,input_size=[384, 576],n_embed=512)',
    "VQ_FastFlow_data_new_2": 'build_VQ_FastFlow(pretrain_path="/mnt/kamiya/code/VQ-FastFlow/_fastflow_experiment_checkpoints/exp_dn_data_new_2_K512/best.pt",backbone_name="wide_resnet50_2",flow_steps=8,input_size=[384, 576],n_embed=512)',
    "VQ_FastFlow_data_new_3": 'build_VQ_FastFlow(pretrain_path="/mnt/kamiya/code/VQ-FastFlow/_fastflow_experiment_checkpoints/exp_dn_data_new_3_K512_freeze/139.pt",backbone_name="wide_resnet50_2",flow_steps=8,input_size=[384, 576],n_embed=512)',
    "VQ_FastFlow_data3": 'build_VQ_FastFlow(pretrain_path="/mnt/kamiya/code/VQ-FastFlow/_fastflow_experiment_checkpoints/exp_dn_data3_K128/best.pt", backbone_name="wide_resnet50_2",flow_steps=8,input_size=[320, 480],n_embed=128)',
    "VQ_FastFlow_data4": 'build_VQ_FastFlow(pretrain_path="/mnt/kamiya/code/VQ-FastFlow/_fastflow_experiment_checkpoints/exp_dn_data4_K128/best.pt", backbone_name="wide_resnet50_2",flow_steps=8,input_size=[320, 480],n_embed=128)'
}

def load(name):
    return eval(_BACKBONES[name])
