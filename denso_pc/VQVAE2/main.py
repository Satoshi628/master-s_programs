import argparse
import os
import re

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
import yaml
# from ignite.contrib import metrics
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, compute_pro
import constants as const
import dataset
import vqvae
import sqvae
import utils
from visual import show_quant_indices, show_anomaly
import matplotlib.pyplot as plt

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


def build_model(args, config):
    print(args.model)
    if args.model == "VQVAE":
        model = vqvae.VQVAE2(
            backbone_name=config["backbone_name"],
            input_size=config["input_size"],
            n_embed=config["n_embed"]
        )
    if args.model == "SQVAE":
        model = sqvae.Gaussian_SQVAE(
            backbone_name=config["backbone_name"],
            input_size=config["input_size"],
            n_embed=config["n_embed"]
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
        loss = ret["quantize_loss"] + ret["reconstraction_loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        del loss
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f}) quantize loss = {:.3f} recon loss = {:.3f}".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg, ret["quantize_loss"].item(), ret["reconstraction_loss"].item()
                )
            )
    return loss_meter.avg

@torch.no_grad()
def eval_once(args, dataloader, model):
    model.eval()
    quant_num = yaml.safe_load(open(args.config, "r"))["n_embed"]

    idx = 0
    quantize_loss = utils.AverageMeter()
    recon_loss = utils.AverageMeter()
    for data, targets in tqdm(dataloader, total=len(dataloader)):
        data, targets = data.cuda(), targets.cuda()
        ret = model(data)
        # log
        quantize_loss.update(ret["quantize_loss"].item())
        recon_loss.update(ret["reconstraction_loss"].item())

        recon_imgs = torch.cat([data, ret["images"]],dim=0)
        torchvision.utils.save_image(recon_imgs,
                                f"{const.CHECKPOINT_DIR}/exp_{args.category}_K{quant_num}/recon_imgs_{idx:04}.png",
                                normalize=True,
                                n_row=data.shape[0])
        show_quant_indices(ret["quant_indices"],
            f"{const.CHECKPOINT_DIR}/exp_{args.category}_K{quant_num}/quant_indices_{idx:04}.png",
            quant_num=quant_num)
        idx += 1
    
    return {"quantize_loss": quantize_loss.avg, "reconstraction_loss": recon_loss.avg}


def train(args):
    config = yaml.safe_load(open(args.config, "r"))
    quant_num = config["n_embed"]
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, f"exp_{args.category}_K{quant_num}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = build_model(args, config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    best_score = 999999.
    best_epoch = 0

    for epoch in range(const.NUM_EPOCHS):
        train_loss = train_one_epoch(train_dataloader, model, optimizer, epoch)
        if best_score > train_loss:
            best_epoch = epoch
            best_score = train_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "best.pt"),
            )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(args, config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_loss_dict = eval_once(args, test_dataloader, model)
    
    quant_num =config["n_embed"]
    result_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}_K{quant_num}", "result.txt")
    with open(result_path, "w") as f:
        for k, v in eval_loss_dict.items():
            f.write(f"{k}\t{v:.3f}\n")

#どの量子ベクトルがどれだけ使われているかのヒストグラム
@torch.no_grad()
def visualization(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(args, config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    model.eval()
    quant_num =config["n_embed"]
    
    quant_indices_dict = {"layer1":[],"layer2":[],"layer3":[]}
    quant_label_dict = {"layer1":[],"layer2":[],"layer3":[]}
    
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        data = data.cuda()
        ret = model(data)

        quant = ret["quant_indices"][0].cpu().numpy()
        quant_indices_dict["layer1"].append(quant)
        quant_label_dict["layer1"].append(np.zeros_like(quant))
        
        quant = ret["quant_indices"][1].cpu().numpy()
        quant_indices_dict["layer2"].append(quant)
        quant_label_dict["layer2"].append(np.zeros_like(quant))

        quant = ret["quant_indices"][2].cpu().numpy()
        quant_indices_dict["layer3"].append(quant)
        quant_label_dict["layer3"].append(np.zeros_like(quant))

    for data, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        data, targets = data.cuda(), targets.cuda()
        ret = model(data)

        quant = ret["quant_indices"][0].cpu().numpy()
        masks = F.interpolate(targets, size=quant.shape[-2:])[:, 0]
        masks = (masks > 0.5)*1.0
        quant_indices_dict["layer1"].append(quant)
        quant_label_dict["layer1"].append(masks.cpu().numpy())
        
        quant = ret["quant_indices"][1].cpu().numpy()
        masks = F.interpolate(targets, size=quant.shape[-2:])[:, 0]
        masks = (masks > 0.5)*1.0
        quant_indices_dict["layer2"].append(quant)
        quant_label_dict["layer2"].append(masks.cpu().numpy())

        quant = ret["quant_indices"][2].cpu().numpy()
        masks = F.interpolate(targets, size=quant.shape[-2:])[:, 0]
        masks = (masks > 0.5)*1.0
        quant_indices_dict["layer3"].append(quant)
        quant_label_dict["layer3"].append(masks.cpu().numpy())
    
    for k in quant_indices_dict.keys():
        quant = np.concatenate(quant_indices_dict[k], axis=0)
        label = np.concatenate(quant_label_dict[k], axis=0)

        normaly = quant[label == 0]
        anomaly = quant[label == 1]

        bins = quant_num
        # plt.hist([normaly, anomaly], bins, range=[0, 511], color=["b", "r"], label=['normaly', 'anomaly'], log=True, stacked=True)
        plt.hist(normaly, bins, alpha = 0.5, color="b", label='normaly', log=True)
        plt.hist(anomaly, bins, alpha = 0.5, color="r", label='anomaly', log=True)
        plt.legend(loc='upper left')
        #f"{const.CHECKPOINT_DIR}/exp_{args.category}_K{quant_num}/recon_imgs_{idx:04}.png"
        plt.savefig(f"{const.CHECKPOINT_DIR}/exp_{args.category}_K{quant_num}/quantize_hist_{k}.png")
        plt.close()
        normaly_idx, normaly_counts = np.unique(normaly, return_counts=True)
        anomaly_idx, anomaly_counts = np.unique(anomaly, return_counts=True)
        counts = np.zeros([2, 512])
        counts[0, normaly_idx] = normaly_counts
        counts[1, anomaly_idx] = anomaly_counts
        counts = counts[:, np.all(counts != 0, axis=0)]
        # counts /= len(normaly)
        plt.scatter(counts[0], counts[1], s=10, color="k")
        plt.savefig(f"{const.CHECKPOINT_DIR}/exp_{args.category}_K{quant_num}/scatter_{k}.png")
        print(f"{k}の共分散:{np.cov(counts[0], counts[1])}")
        plt.close()


#量子ベクトルを正常品で使われているベクトルのみ残し検証する
@torch.no_grad()
def reduce_emb(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(args, config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    model.eval()

    quant_num =config["n_embed"]
    result_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}_K{quant_num}", "result_by_reduce_quant.txt")
    with open(result_path, "w") as f:
        pass


    quant_indices_dict = {"layer1":[],"layer2":[],"layer3":[]}
    quant_label_dict = {"layer1":[],"layer2":[],"layer3":[]}
    
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        data = data.cuda()
        ret = model(data)

        quant = ret["quant_indices"][0].cpu().numpy()
        quant_indices_dict["layer1"].append(quant)
        quant_label_dict["layer1"].append(np.zeros_like(quant))
        
        quant = ret["quant_indices"][1].cpu().numpy()
        quant_indices_dict["layer2"].append(quant)
        quant_label_dict["layer2"].append(np.zeros_like(quant))

        quant = ret["quant_indices"][2].cpu().numpy()
        quant_indices_dict["layer3"].append(quant)
        quant_label_dict["layer3"].append(np.zeros_like(quant))
    
    for idx, k in enumerate(quant_indices_dict.keys()):
        quant = np.concatenate(quant_indices_dict[k], axis=0)
        label = np.concatenate(quant_label_dict[k], axis=0)

        normaly = quant.reshape(-1)
        normaly_idx, normaly_counts = np.unique(normaly, return_counts=True)
        counts = np.zeros([512])
        counts[normaly_idx] = normaly_counts
        # counts = counts[counts != 0]
        model.vq_list[idx].embed = model.vq_list[idx].embed[:,counts != 0]
        print(f"{k} quantize number:{quant_num} => {model.vq_list[idx].embed.shape[1]}")
        
        with open(result_path, "a") as f:
            f.write(f"{k} quantize number:{quant_num} => {model.vq_list[idx].embed.shape[1]}\n")
    
    # test
    quantize_loss = utils.AverageMeter()
    recon_loss = utils.AverageMeter()
    

    for data, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        data, targets = data.cuda(), targets.cuda()
        ret = model(data)
        # log
        quantize_loss.update(ret["quantize_loss"].item())
        recon_loss.update(ret["reconstraction_loss"].item())
    result = {"quantize_loss": quantize_loss.avg, "reconstraction_loss": recon_loss.avg}

    
    with open(result_path, "a") as f:
        for k, v in result.items():
            f.write(f"{k}\t {v:.3f}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument("--model", type=str, required=True, choices=["VQVAE", "SQVAE"], help="choice model")
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
    parser.add_argument("--reduce", action="store_true", help="reduce")
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"category:{args.category}")
    # visualization(args)
    if args.eval:
        if args.reduce:
            reduce_emb(args)
        else:
            evaluate(args)
    else:
        train(args)
