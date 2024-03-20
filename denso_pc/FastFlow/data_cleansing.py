import argparse
import os
import re
import shutil

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import yaml
# from ignite.contrib import metrics
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, compute_pro
import constants as const
import dataset
import fastflow
import utils
from visual import show_anomaly
import matplotlib.pyplot as plt
from otsu_threshold import otsu_threshold
from sklearn.mixture import GaussianMixture

def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset_Filter(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model

@torch.no_grad()
def make_data(args, dataloader, model):
    model.eval()
    images = []
    predicts = []
    image_names = []
    
    mask = Image.open(os.path.join(args.data, args.category, "mask.png")).convert("L")
    mask = transforms.Compose(
        [
            transforms.Resize(yaml.safe_load(open(args.config, "r"))["input_size"]),
            transforms.ToTensor(),
        ]
    )(mask)[None]

    # save_anomaly_path = os.path.join(args.save_path, "no_use_anomaly_maps")
    # if not os.path.isdir(save_anomaly_path):
    #     os.makedirs(save_anomaly_path)
    # auroc_metric = metrics.ROC_AUC()
    for data, file_name in tqdm(dataloader, total=len(dataloader)):
        data = data.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs[(mask == 0).expand(outputs.shape[0], -1, -1, -1)] = -1.

        images.append(data.cpu().detach().numpy())
        image_names.extend(file_name)
        predicts.append(outputs.numpy())
    
    images = np.concatenate(images, axis=0)
    predicts = np.concatenate(predicts, axis=0)

    per_image = predicts.reshape(predicts.shape[0], -1).max(axis=-1)

    # th = otsu_threshold(per_image)
    # GMM = GaussianMixture(2, verbose=1)
    # GMM.fit(predicts.flatten()[:, None])
    # classes = GMM.predict(predicts.flatten()[:, None])

    # n, _, _ = plt.hist(per_image, bins=100, alpha = 0.5, color="b")
    # plt.vlines(th, 0, max(n), "k")
    
    #全体の1%を異常とする
    p = 0.05
    p_th = np.sort(per_image)[::-1][int(per_image.shape[0]*p)]
    # plt.vlines(p_th, 0, max(n), "r")
    
    # plt.savefig(f"image_hist.png")
    # plt.close()
    flag =  per_image > p_th
    ano_images = images[flag]
    ano_predicts = predicts[flag]
    ano_image_names = [name for name, f in zip(image_names, flag) if f]
    # show_anomaly(ano_images, ano_predicts, ano_image_names, save_anomaly_path)

    image_names = [name for name, f in zip(image_names, flag) if not f]

    save_dir = os.path.join(args.data, args.category, f"train_clean_{p:.2f}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "good"))
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "good"))

    save_dir = os.path.join(save_dir, "good")

    for name in tqdm(image_names):
        new_name = name.replace("train", f"train_clean_{p:.2f}")
        shutil.copy(name, new_name)

def neural_style_transfer(args, dataloader, model):
    model.eval()
    input_size = yaml.safe_load(open(args.config, "r"))["input_size"]

    img_path = "/mnt/kamiya/dataset/DENSO_D/dn_data_new/test/broken/190307083546.png"
    normal_path = "/mnt/kamiya/dataset/DENSO_D/dn_data_new/train/good/190307083400.png"

    img = Image.open(img_path).convert("RGB")
    base_img = Image.open(normal_path).convert("RGB")
    trans = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    img = trans(img)
    base_img = trans(base_img)
    ori_img = img[None].clone()
    data = img[None].clone().detach().cuda().requires_grad_()
    base_img = base_img[None].clone().detach().cuda().requires_grad_()

    optimizer = torch.optim.Adam([data], lr=0.003)
    rank = 20
    with tqdm(range(2000)) as pbar:
        for _ in pbar:
            ret = model(data)
            outputs = ret["anomaly_map"]
            outputs = outputs.flatten()
            values, indices = torch.topk(outputs, k=rank)
            loss = values.mean() + ((outputs[indices] - base_img.flatten()[indices])**2).mean()
            # loss = outputs.mean()
            pbar.set_description(f"[loss {loss:.3f}]")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    normal_img = data.clone().detach().cpu()
    group = torch.cat([ori_img, normal_img], dim=0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    group = group*std[None, :, None, None] + mean[None, :, None, None]
    group = torch.cat([group, group[:1]-group[1:]],dim=0)
    torchvision.utils.save_image(group,
                                    "test_style_image.png",
                                    #nrow=1,
                                    normalize=True)


def optimal_color(args, dataloader, model):
    model.eval()
    input_size = yaml.safe_load(open(args.config, "r"))["input_size"]

    img_path1 = "/mnt/kamiya/dataset/DENSO_D/dn_data_new/train/good/190307081957.png"
    img_path2 = "/mnt/kamiya/dataset/DENSO_D/dn_data_new/train/good/190409022313.png"

    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
    trans = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    img1 = trans(img1)
    img2 = trans(img2)
    W = img1.shape[-1]
    img = torch.cat([img1[:,:,:W//2], img2[:,:,W//2:]], dim=-1)
    torchvision.utils.save_image(torch.stack([img1,img2,img]),
                                "half_img.png",
                                #nrow=1,
                                normalize=True)



def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    # neural_style_transfer(args, test_dataloader, model)
    make_data(args, test_dataloader, model)
    # optimal_color(args, test_dataloader, model)




def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument("--save_path", type=str, required=True, help="path to save data folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"category:{args.category}")

    evaluate(args)
