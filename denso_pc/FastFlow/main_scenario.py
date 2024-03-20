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
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, eval_seg_pro, compute_AUAnomaly_Curve
import constants as const
import dataset
import fastflow
import utils
from visual import show_anomaly
from dataset import _CLASSNAMES, _SCENARIO


def build_train_data_loader(args, config, broken):
    train_dataset = dataset.MVTecDataset_Scenario(
        root=args.data,
        category=args.category,
        scenario=args.scenario,
        broken_type=broken,
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


def build_test_data_loader(args, config, broken):

    test_dataset = dataset.MVTecDataset_Scenario(
        root=args.data,
        category=args.category,
        scenario=args.scenario,
        broken_type=broken,
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
        data = data[1].cuda()
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


def eval_once(config, dataloader, model):
    model.eval()
    anomaly_types = []
    predicts = []
    ano_maps = []
    labels = []

    idx = 0
    anoma_max = 0.
    anoma_min = 1.
    # auroc_metric = metrics.ROC_AUC()
    for anomaly, data, targets in tqdm(dataloader, total=len(dataloader)):
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        anomaly_maps = ret["anomaly_maps"].cpu().detach()

        # if outputs.max() > anoma_max : anoma_max = outputs.max().item()
        # if outputs.min() < anoma_min : anoma_min = outputs.min().item()
        
        # show_anomaly(data, targets, outputs, f"candle/anomaly{idx}.png", limit=[anoma_min, anoma_max])
        # idx += 1
        anomaly_types.extend(anomaly)
        predicts.append(outputs.numpy())
        ano_maps.append(anomaly_maps.numpy())
        targets = (targets != 0) * 1.0
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
    PRO = eval_seg_pro(pixel_labels, pixel_scores)

    if dataloader.dataset.scenario == "A2N":
        anomaly_flag = {dataloader.dataset.broken_type:[dataloader.dataset.broken_type in ano for ano in anomaly_types]}
    elif dataloader.dataset.scenario == "N2A":
        anomaly_flag = {"pseudo_anomaly":["pseudo_anomaly" in ano for ano in anomaly_types]}
    elif dataloader.dataset.scenario == "Normal" and "good-pseudo_anomaly" in anomaly_types: # N2A Normal
        anomaly_flag = {"pseudo_anomaly":["pseudo_anomaly" in ano for ano in anomaly_types]}
    else: #A2N Normal
        anomaly_flag = {broken_type:[broken_type in ano for ano in anomaly_types] for broken_type in _SCENARIO[dataloader.dataset.category]}

    
    OnlyL1Score = {}
    OtherL1Score = {}
    for broken_type, flag in anomaly_flag.items():
        flag = np.array(flag)
        image_labels = np.array(image_labels)
        OnlyL1Score[broken_type + "-OnlyL1Score"] = compute_AUAnomaly_Curve(image_scores[flag])
        OtherL1Score[broken_type + "-OtherL1Score"] = compute_imagewise_retrieval_metrics(image_scores[~flag], image_labels[~flag])["auroc"]

    print("Image level AUROC: {:.2%}".format(imagewize_AUROC["auroc"]))
    print("Pixel level AUROC: {:.2%}".format(pixelwize_AUROC["auroc"]))
    print("PRO: {:.2%}".format(PRO))
    return {"Image_level_AUROC":imagewize_AUROC["auroc"], "Pixel_level_AUROC":pixelwize_AUROC["auroc"], "PRO": PRO, **OnlyL1Score, **OtherL1Score}


def train(args):
    if args.scenario == "Normal":
        broken_type = ["None"]
    elif args.scenario == "A2N":
        broken_type = _SCENARIO[args.category]
    elif args.scenario == "N2A":
        broken_type = ["pseudo_anomaly"]
    
    for broken in broken_type:

        os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_dir = os.path.join(
            const.CHECKPOINT_DIR, f"_{args.scenario}-exp_{args.category}-{broken}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        config = yaml.safe_load(open(args.config, "r"))
        model = build_model(config)
        optimizer = build_optimizer(model)

        train_dataloader = build_train_data_loader(args, config, broken)
        test_dataloader = build_test_data_loader(args, config, broken)
        model.cuda()


        best_score = 0.0
        best_epoch = 0
        best_result = {"temp",0.0}
        for epoch in range(const.NUM_EPOCHS):
            train_one_epoch(train_dataloader, model, optimizer, epoch)
            if (epoch + 1) % const.EVAL_INTERVAL == 0:
                eval_score = eval_once(config, test_dataloader, model)
                
                if best_score < np.mean(list(eval_score.values())[:3]):
                    best_epoch = epoch
                    best_score = np.mean(list(eval_score.values())[:3])
                    best_result = eval_score
                torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(checkpoint_dir, "best.pt"),
                    )

        
        result_path = os.path.join(const.CHECKPOINT_DIR, f"_{args.scenario}-exp_{args.category}-{broken}", "_best_epoch.txt")
        with open(result_path, "w") as f:
            f.write(f"best epoch: {best_epoch:d}")
        
        
        result_path = os.path.join(const.CHECKPOINT_DIR, f"_{args.scenario}-exp_{args.category}-{broken}", "_result.txt")
        with open(result_path, "w") as f:
            for k, v in best_result.items():
                f.write(f"{k}\t {v:.2%}\n")

def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    result_dict = eval_once(test_dataloader, model)
    
    result_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}-{broken}", "_result.txt")
    with open(result_path, "w") as f:
        for k, v in result_dict.items():
            f.write(f"{k}\t {v:.2%}\n")


@torch.no_grad()
def vis_std(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    model.eval()
    
    output_maps = []
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        data = data.cuda()
        ret = model(data)
        output_map = ret["output_map"].cpu().detach()
        output_maps.append(output_map.numpy())
    
    images = data[:1].cpu()

    predicts = []
    labels = []
    for data, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        data, targets = data.cuda(), targets.cuda()
        ret = model(data)

        outputs = ret["anomaly_map"].cpu().detach()
        predicts.append(outputs.numpy())
        labels.append(targets.cpu().detach().numpy())
    
    predicts = np.concatenate(predicts, axis=0)
    output_maps = np.concatenate(output_maps, axis=0)
    labels = np.concatenate(labels, axis=0)

    H,W = output_maps.shape[-2:]
    normal_img_mean = np.mean(output_maps, axis=0)
    normal_img_std = np.std(output_maps, axis=0)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    images = images*std[None, :, None, None] + mean[None, :, None, None]

    #平均可視化
    anomaly = torch.from_numpy(normal_img_mean)

    anomaly_max = anomaly.max()
    anomaly_min = anomaly.min()
    anomaly = (anomaly-anomaly_min) / (anomaly_max - anomaly_min + 1e-8)

    #convert color jet
    anomaly = anomaly.detach().cpu().numpy() *255
    anomaly = anomaly.astype(np.uint8)
    anomaly = cv2.applyColorMap(anomaly, cv2.COLORMAP_JET)
    anomaly = torch.from_numpy(anomaly).float() /255.
    anomaly = anomaly.permute(2,0,1)[[2,1,0]]

    rate = 0.7
    anomaly = images*rate + anomaly[None]*(1-rate)
    groups = torch.cat([images, anomaly], dim=0)
    torchvision.utils.save_image(groups,
                                f"images/{args.category}_mean_image_max{anomaly_max.item():.2f}_min{anomaly_min.item():.2f}.png",
                                normalize=True)
    #標準偏差可視化
    anomaly = torch.from_numpy(normal_img_std)


    anomaly_max = anomaly.max()
    anomaly_min = anomaly.min()
    anomaly = (anomaly-anomaly_min) / (anomaly_max - anomaly_min + 1e-8)

    #convert color jet
    anomaly = anomaly.detach().cpu().numpy() *255
    anomaly = anomaly.astype(np.uint8)
    anomaly = cv2.applyColorMap(anomaly, cv2.COLORMAP_JET)
    anomaly = torch.from_numpy(anomaly).float() /255.
    anomaly = anomaly.permute(2,0,1)[[2,1,0]]

    rate = 0.7
    anomaly = images*rate + anomaly[None]*(1-rate)
    groups = torch.cat([images, anomaly], dim=0)
    torchvision.utils.save_image(groups,
                                f"images/{args.category}_std_image_max{anomaly_max.item():.2f}_min{anomaly_min.item():.2f}.png",
                                normalize=True)

    #異常箇所の出やすい所可視化
    anomaly_prob = labels.mean(axis=0)
    
    anomaly_max = anomaly_prob.max()
    anomaly_min = anomaly_prob.min()
    anomaly_prob = (anomaly_prob-anomaly_min) / (anomaly_max - anomaly_min + 1e-8)
    anomaly_prob = anomaly_prob * 255
    anomaly_prob = anomaly_prob.astype(np.uint8)
    anomaly_prob = cv2.applyColorMap(anomaly_prob[0], cv2.COLORMAP_JET)
    anomaly_prob = torch.from_numpy(anomaly_prob).float() /255.
    anomaly_prob = anomaly_prob.permute(2,0,1)[[2,1,0]]

    rate = 0.7
    anomaly_prob = images*rate + anomaly_prob*(1-rate)
    groups = torch.cat([images, anomaly_prob], dim=0)
    torchvision.utils.save_image(groups,
                                f"images/{args.category}_anomaly_prob_image_max{anomaly_max.item():.2f}_min{anomaly_min.item():.2f}.png",
                                normalize=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument("--scenario", type=str, required=True, help="path to mvtec folder")
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
    # vis_std(args)
    if args.eval:
        evaluate(args)
    else:
        train(args)
