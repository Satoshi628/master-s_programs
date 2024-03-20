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


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        del loss
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )
    return loss_meter

def eval_once(dataloader, model):
    model.eval()
    predicts = []
    ano_maps = []
    labels = []

    idx = 0
    anoma_max = 0.
    anoma_min = 1.
    # auroc_metric = metrics.ROC_AUC()
    for data, targets in tqdm(dataloader, total=len(dataloader)):
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        anomaly_maps = ret["anomaly_maps"].cpu().detach()

        if outputs.max() > anoma_max : anoma_max = outputs.max().item()
        if outputs.min() < anoma_min : anoma_min = outputs.min().item()
        
        # show_anomaly(data, targets, outputs, f"anomaly{idx}.png", limit=[anoma_min, anoma_max])
        idx += 1
        predicts.append(outputs.numpy())
        ano_maps.append(anomaly_maps.numpy())
        labels.append(targets.cpu().detach().numpy())
    
    predicts = np.concatenate(predicts, axis=0)
    ano_maps = np.concatenate(ano_maps, axis=0)
    labels = np.concatenate(labels, axis=0)

    #ano_maps = ano_maps[:,:,:,:,0]
    image_scores = predicts.reshape(predicts.shape[0], -1).max(axis=-1)
    image_labels = labels.reshape(labels.shape[0], -1).max(axis=-1)
    pixel_scores = predicts[:,0]
    pixel_labels = labels[:,0]

    imagewize_AUROC = compute_imagewise_retrieval_metrics(image_scores, image_labels)
    pixelwize_AUROC = compute_pixelwise_retrieval_metrics(pixel_scores, pixel_labels)
    PRO = compute_pro(pixel_labels, pixel_scores)

    print("Image level AUROC: {:.2%}".format(imagewize_AUROC["auroc"]))
    print("Pixel level AUROC: {:.2%}".format(pixelwize_AUROC["auroc"]))
    print("PRO: {:.2%}".format(PRO))

    return {"Image_level_AUROC":imagewize_AUROC["auroc"], "Pixel_level_AUROC":pixelwize_AUROC["auroc"], "PRO": PRO}


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    txt_path = os.path.join(checkpoint_dir, "training.txt")
    with open(txt_path, "w") as f:
        f.write("train_loss\tImage_level_AUROC\tPixel_level_AUROC\tPRO\tAverage\n")

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    best_loss = 9999999.
    best_score = 0.0

    for epoch in range(const.NUM_EPOCHS):
        loss_meter = train_one_epoch(train_dataloader, model, optimizer, epoch)

        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_score = eval_once(test_dataloader, model)
            
            if best_score < np.mean(list(eval_score.values())):
                best_score = np.mean(list(eval_score.values()))
                torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(checkpoint_dir, "best_score.pt"),
                    )
            if loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
                torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(checkpoint_dir, "best_loss.pt"),
                    )
            
            with open(txt_path, "a") as f:
                text = f"{loss_meter.avg:.4f}\t" + "\t".join([f"{v:.2%}" for v in eval_score.values()]) + \
                    f"\t{np.mean(list(eval_score.values())):.2%}\n"
                f.write(text)

def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    result_dict = eval_once(test_dataloader, model)
    
    best_type = os.path.splitext(os.path.basename(args.checkpoint))[0]
    result_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}", f"_{best_type}_result.txt")
    with open(result_path, "w") as f:
        for k, v in result_dict.items():
            f.write(f"{k}\t {v:.2%}\n")


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
        "-ckpt", "--checkpoint", type=str, default="", help="path to load checkpoint"
    )
    args = parser.parse_args()

    args.checkpoint = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}", args.checkpoint)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"category:{args.category}")
    # vis_std(args)
    if args.eval:
        evaluate(args)
    else:
        train(args)
