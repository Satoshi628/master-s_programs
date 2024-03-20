#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
import logging
from logging import getLogger

#----- 専用ライブラリ -----#
from tqdm import tqdm
import hydra
import numpy as np
import torch
import torch.nn as nn
import torchvision
from anomalib.utils.metrics import AUPRO, AUROC
import patchcore
from patchcore.patchcore import PatchCore
from patchcore.metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics

#----- 自作モジュール -----#
from utils.dataset import MVtecAD_Loader
import utils.transform as tf
from utils.visualization import feature_Segmentation
from models.SegCore import SegCore
from models.sampler import KNN_sampler, Balance_sampler

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
logger.info('message')

def build_transform(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.Numpy2Tensor(),  # numpy to Tensor
                                    tf.Resize(size=cfg.dataset.resize),
                                    tf.CenterCrop(size=cfg.dataset.crop),
                                    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    test_transform = tf.Compose([tf.Numpy2Tensor(),    # numpy to Tensor
                                tf.Resize(size=cfg.dataset.resize),
                                tf.CenterCrop(size=cfg.dataset.crop),
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])
    
    return train_transform, test_transform

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform, test_transform = build_transform(cfg)

    if cfg.dataset.product == "ALL":
        products = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
    elif cfg.dataset.product == "Texture":
        products = ["carpet", "grid", "leather", "tile", "wood"]
    elif cfg.dataset.product == "Object":
        products = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
    else:
        products = [cfg.dataset.product]
    
    train_dataloader_dict = {}
    test_dataloader_dict = {}
    for pro in products:
        train_dataset = MVtecAD_Loader(root_dir=cfg.dataset.root_dir, product=pro, dataset_type='train', transform=train_transform)
        test_dataset = MVtecAD_Loader(root_dir=cfg.dataset.root_dir, product=pro, dataset_type='test', transform=test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=cfg.exp.batch_size,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    num_workers=2,
                                                    pin_memory=True)
        
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=cfg.exp.batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=2,
                                                pin_memory=True)
        train_dataloader_dict[pro] = train_loader
        test_dataloader_dict[pro] = test_loader

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_dataloader_dict, test_dataloader_dict


############## evalation関数 ##############
def evalation(model, train_dataloader_dict, test_dataloader_dict, device):
    # モデル→推論モード
    model.eval()

    all_metrics = {}

    for k in train_dataloader_dict.keys():
        score_dict = {"scores": [], "segmentations": []}

        train_loader = train_dataloader_dict[k]
        test_loader = test_dataloader_dict[k]
        
        all_masks = []
        all_targets = []

        model.fit(train_loader)

        # feature_Segmentation(model.get_similar_map(), k)

        for batch_idx, (inputs, masks, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            all_masks.extend([m for m in masks.numpy()])
            all_targets.extend([t for t in targets.numpy()])

            image_level, pixel_level = model.predict(inputs)
            score_dict["scores"].extend(image_level)
            score_dict["segmentations"].extend(pixel_level)
        
        scores = np.array(score_dict["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(score_dict["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        auroc = compute_imagewise_retrieval_metrics(scores, all_targets)["auroc"]

        # Compute PRO score & PW Auroc for all images
        pixel_scores = compute_pixelwise_retrieval_metrics(
            segmentations, all_masks
        )
        full_pixel_auroc = pixel_scores["auroc"]

        # Compute PRO score & PW Auroc only images with anomalies
        sel_idxs = []
        for i in range(len(all_masks)):
            if np.sum(all_masks[i]) > 0:
                sel_idxs.append(i)
        pixel_scores = compute_pixelwise_retrieval_metrics(
            [segmentations[i] for i in sel_idxs],
            [all_masks[i] for i in sel_idxs],
        )
        anomaly_pixel_auroc = pixel_scores["auroc"]

        metrics = {
            'AUROC-image':auroc,
            'AUROC-pixel':full_pixel_auroc,
            'AUPRO-pixel': anomaly_pixel_auroc
        }

        logger.info(f'{k} TEST: AUROC-image: {auroc:.2%} | AUROC-pixel: {full_pixel_auroc:.2%} | AUPRO-pixel: {anomaly_pixel_auroc:.2%}')
        all_metrics[k] = [auroc, full_pixel_auroc, anomaly_pixel_auroc]

    return all_metrics

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    
    logger.info(f"gpu         :{cfg.exp.gpu}")
    logger.info(f"batch size  :{cfg.exp.batch_size}")
    logger.info(f"product     :{cfg.dataset.product}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.exp.gpu) if torch.cuda.is_available() else 'cpu')

    # model = SegCore("wideresnet50", cfg.dataset.crop, device=device).cuda(device)
    model = SegCore(device)

    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.1, device)
    # sampler = KNN_sampler(0.1, device)
    sampler = Balance_sampler(0.1, device)
    model.load(
        backbone_name="wideresnet50",
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=[3] + cfg.dataset.crop,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        featuresampler=sampler
    )


    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader, val_loader = dataload(cfg)

    # evalation
    start = time.time()
    matrics = evalation(model, train_loader, val_loader, device)
    end = time.time()

    PATH = "result.txt"
    with open(PATH, mode='w') as f:
        f.write("category,Image level AUROC, Pixel level AUROC,AUPRO\n")
        for k, v in matrics.items():
            f.write(f"{k},{v[0]:.1%},{v[1]:.1%},{v[2]:.1%}\n")
    
    run_time = end - start
    with open("time.txt", mode='w') as f:
        f.write("{:.8f}\n".format(run_time))


############## main ##############
if __name__ == '__main__':
    main()