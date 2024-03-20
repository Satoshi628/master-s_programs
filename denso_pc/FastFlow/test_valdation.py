import argparse
import os
import re

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision
import yaml
# from ignite.contrib import metrics
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, compute_pro
import constants as const
import dataset
import fastflow
import utils
from visual import show_anomaly

def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
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

def see_label(label, image_name):  #C,H,W
    label = label.cpu()

    label = label.numpy().transpose(1,2,0)
    label = label* 255
    label=label.astype('uint8')
    
    image_name = os.path.split(image_name)[-1]
    cv2.imwrite("test_val_label_datanew2/" + f'{image_name}', label)

def see_img_heatmap(data,segresult,image_name):  #C,H,W
    data = data.cpu()
    segresult = segresult[0].cpu()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean, std = torch.tensor(mean, dtype=torch.float), torch.tensor(std, dtype=torch.float)
    
    data = data*std[:,None, None] + mean[:,None, None]
    y2max = 255
    y2min = 0
    segresult = segresult.numpy()
    x2max = segresult.max()
    x2min = segresult.min()
    segresult = np.round((y2max - y2min) * (segresult - x2min) / (x2max - x2min) + y2min)
    segresult = segresult.astype(np.uint8)
    heatmap = cv2.applyColorMap(segresult, colormap=cv2.COLORMAP_JET)
    alpha = 0.15
    alpha2 = 0.3
    
    data=data.permute(1,2,0)
    data = data.numpy()
    data = data* 255
    data=data.astype('uint8')
    overlay = data.copy()
    data = cv2.addWeighted(heatmap, alpha2, overlay, 1 - alpha, 0, overlay)
    
    image_name = os.path.split(image_name)[-1]
    cv2.imwrite("test_val_data4/" + f'{image_name}', data)


@torch.no_grad()
def eval_once(dataloader, model):
    model.eval()
    predicts = []
    labels = []
    image_names = []
    # auroc_metric = metrics.ROC_AUC()
    for data, targets, image_file in tqdm(dataloader, total=len(dataloader)):
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()

        #visualization
        for label, img_name in zip(targets, image_file):
            see_label(label, img_name)

        # for img, ano_map, img_name in zip(data, outputs, image_file):
        #     see_img_heatmap(img, ano_map, img_name)

        image_names.extend(image_file)
        predicts.append(outputs.numpy())
        labels.append(targets.cpu().detach().numpy())
    
    predicts = np.concatenate(predicts, axis=0)
    labels = np.concatenate(labels, axis=0)

    image_scores = predicts.reshape(predicts.shape[0], -1).max(axis=-1)
    image_scores = (image_scores - image_scores.min())/ (image_scores.max() - image_scores.min())
    image_labels = labels.reshape(labels.shape[0], -1).max(axis=-1)

    res_score = image_labels - image_scores
    sort_idx = np.argsort(image_names)
    
    image_names = [image_names[idx] for idx in sort_idx]
    image_scores = image_scores[sort_idx]
    image_labels = image_labels[sort_idx]
    res_score = res_score[sort_idx]

    return image_names, image_scores, image_labels, res_score


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    
    image_names, image_scores, image_labels, res_score = eval_once(test_dataloader, model)

    result_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}", "_test.txt")
    with open(result_path, "w") as f:
        f.write(f"{args.category}\n")
        for path, score, label, res in zip(image_names, image_scores, image_labels, res_score):
            path = os.path.split(path)[-1]
            f.write(f"{path}\t{score}\t{label}\t{res}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()

    # best_epoch_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}", "_best_epoch.txt")
    # if os.path.exists(best_epoch_path):
    #     with open(best_epoch_path, "r") as f:
    #         best_epoch = re.sub(r"\D", "", f.read())
    #     print(args.checkpoint)
    #     args.checkpoint = os.path.join(*args.checkpoint.split("/")[:-1], f"{best_epoch}.pt")
    #     print(args.checkpoint)
    args.checkpoint = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}", "best.pt")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"category:{args.category}")
    evaluate(args)