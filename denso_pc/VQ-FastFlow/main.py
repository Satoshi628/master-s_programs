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
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, compute_pro, pixel_pro, eval_seg_pro
import constants as const
import dataset
import vq_fastflow
import utils
from visual import show_quant_indices, show_anomaly
import matplotlib.pyplot as plt

def build_train_data_loader(args, config):
    if args.category in const.VISA_CATEGORIES:
        train_dataset = dataset.VisADataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=True,
        )
    else:
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
    if args.category in const.VISA_CATEGORIES:
        test_dataset = dataset.VisADataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=False,
        )
    else:
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
    model = vq_fastflow.VQ_FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        n_embed=config["n_embed"]
    )
    print(config["n_embed"])
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
        # for idx in range(len(model.quantizer)):
        #     temperature_current = np.max([model.quantizer[idx].temperature * np.exp(-0.0001), 0.07])
        #     model.quantizer[idx].set_temperature(temperature_current)
        # quantize_loss = ret["quantize_loss"]
        # jac_dets_loss = ret["jac_dets_loss"]
        # reconstraction_loss = ret["reconstraction_loss"]
        # print(f"quantize_loss:{quantize_loss:.3f}")
        # print(f"jac_dets_loss:{-jac_dets_loss:.3f}")
        # print(f"reconstraction_loss:{reconstraction_loss:.3f}")
        
        loss = ret["jac_dets_loss"] + ret["quantize_loss"] + ret["reconstraction_loss"]
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

@torch.no_grad()
def eval_once(dataloader, model):
    model.eval()
    predicts = []
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
        
        if outputs.max() > anoma_max : anoma_max = outputs.max().item()
        if outputs.min() < anoma_min : anoma_min = outputs.min().item()
        show_anomaly(data, targets, outputs, f"candle/anomaly{idx}.png", limit=[anoma_min, anoma_max])
        idx += 1
        recon_imgs = torch.cat([data, ret["images"]],dim=0)
        # torchvision.utils.save_image(recon_imgs,
        #                         "recon_imgs.png",
        #                         normalize=True,
        #                         n_row=data.shape[0])
        # show_quant_indices(ret["quant_indices"], quant_num=512)
        # input()
        targets = (targets != 0) * 1.0
        predicts.append(outputs.numpy())
        labels.append(targets.cpu().detach().numpy())
    
    predicts = np.concatenate(predicts, axis=0)
    labels = np.concatenate(labels, axis=0)

    image_scores = predicts.reshape(predicts.shape[0], -1).max(axis=-1)
    image_labels = labels.reshape(labels.shape[0], -1).max(axis=-1)
    pixel_scores = predicts[:,0]
    pixel_labels = labels[:,0]

    imagewize_AUROC = compute_imagewise_retrieval_metrics(image_scores, image_labels)
    pixelwize_AUROC = compute_pixelwise_retrieval_metrics(pixel_scores, pixel_labels)
    # PRO = compute_pro(pixel_labels, pixel_scores)
    # PRO2 = pixel_pro(pixel_labels, pixel_scores)
    PRO3 = eval_seg_pro(pixel_labels, pixel_scores)

    print("Image level AUROC: {:.2%}".format(imagewize_AUROC["auroc"]))
    print("Pixel level AUROC: {:.2%}".format(pixelwize_AUROC["auroc"]))
    # print("PRO: {:.2%}".format(PRO))
    # print("PRO2: {:.2%}".format(PRO2))
    print("PRO3: {:.2%}".format(PRO3))

    return {"Image_level_AUROC":imagewize_AUROC["auroc"], "Pixel_level_AUROC":pixelwize_AUROC["auroc"], "PRO": PRO3}


def train(args):
    config = yaml.safe_load(open(args.config, "r"))
    quant_num = config["n_embed"]
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, f"exp_{args.category}_K{quant_num}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    best_score = 0.0
    best_epoch = 0    
    # if best_score < np.mean(list(eval_metrics.values())):
    #     # save best score
    #     state = {'best_step':step}
    #     state.update(eval_log)
    #     json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')
    #     # save best model
    #     torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
    #     _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))
    #     best_score = np.mean(list(eval_metrics.values()))

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_score = eval_once(test_dataloader, model)
            
            if best_score < np.mean(list(eval_score.values())):
                best_epoch = epoch
                best_score = np.mean(list(eval_score.values()))

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
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    result_dict = eval_once(test_dataloader, model)
    
    quant_num =config["n_embed"]
    result_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}_K{quant_num}", "_result.txt")
    with open(result_path, "w") as f:
        for k, v in result_dict.items():
            f.write(f"{k}\t {v:.2%}\n")

@torch.no_grad()
def visualization(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    model.eval()
    
    quant_indices_dict = {"layer1":[],"layer2":[],"layer3":[]}
    quant_label_dict = {"layer1":[],"layer2":[],"layer3":[]}
    
    # for data in tqdm(train_dataloader, total=len(train_dataloader)):
    #     data = data.cuda()
    #     ret = model(data)

    #     quant = ret["quant_indices"][0].cpu().numpy()
    #     quant_indices_dict["layer1"].append(quant)
    #     quant_label_dict["layer1"].append(np.zeros_like(quant))
        
    #     quant = ret["quant_indices"][1].cpu().numpy()
    #     quant_indices_dict["layer2"].append(quant)
    #     quant_label_dict["layer2"].append(np.zeros_like(quant))

    #     quant = ret["quant_indices"][2].cpu().numpy()
    #     quant_indices_dict["layer3"].append(quant)
    #     quant_label_dict["layer3"].append(np.zeros_like(quant))

    mean = torch.tensor([0.485, 0.456, 0.406]).cuda()[None, :,None,None]
    std = torch.tensor([0.229, 0.224, 0.225]).cuda()[None, :,None,None]

    for data, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        data, targets = data.cuda(), targets.cuda()
        ret = model(data)
        data = std*data +mean
        # recon_imgs = torch.cat([data, ret["images"]],dim=0)
        torchvision.utils.save_image(data,
                                "input_images.png",
                                normalize=True,
                                n_row=data.shape[0])
        show_quant_indices(ret["quant_indices"], quant_num=512)
        input("owa")

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
        bins = 128
        under = 0
        # plt.hist([normaly, anomaly], bins, range=[0, 511], color=["b", "r"], label=['normaly', 'anomaly'], log=True, stacked=True)
        plt.hist(normaly, bins, range=[under, under+bins], alpha = 0.5, color="b", label='normaly', log=True)
        plt.hist(anomaly, bins, range=[under, under+bins], alpha = 0.5, color="r", label='anomaly', log=True)
        plt.legend(loc='upper left')
        plt.savefig(f"{k}_quantize_indices.png")
        plt.close()
        normaly_idx, normaly_counts = np.unique(normaly, return_counts=True)
        anomaly_idx, anomaly_counts = np.unique(anomaly, return_counts=True)
        counts = np.zeros([2, 512])
        counts[0, normaly_idx] = normaly_counts
        counts[1, anomaly_idx] = anomaly_counts
        counts = counts[:, np.all(counts != 0, axis=0)]
        # counts /= len(normaly)
        plt.scatter(counts[0], counts[1], s=10, color="k")
        plt.savefig(f"{k}_scatter.png")
        print(f"{k}の共分散:{np.cov(counts[0], counts[1])}")
        plt.close()


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


@torch.no_grad()
def reduce_emb(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    model.eval()

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
        print(model.vq_list[idx].embed.shape)
        
    
    result_dict = eval_once(test_dataloader, model)
    
    quant_num =config["n_embed"]
    result_path = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}_K{quant_num}", "_result.txt")
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
        choices=const.MVTEC_CATEGORIES + const.VISA_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    quant_num = yaml.safe_load(open(args.config, "r"))["n_embed"]
    args.checkpoint = os.path.join(const.CHECKPOINT_DIR, f"exp_{args.category}_K{quant_num}", "best.pt")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"category:{args.category}")
    # visualization(args)
    # reduce_emb(args)
    # vis_std(args)
    if args.eval:
        evaluate(args)
    else:
        train(args)
