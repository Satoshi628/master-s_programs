import os
from time import sleep
import json

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from tensorboard_visualizer import TensorboardVisualizer
from sklearn.metrics import roc_auc_score, average_precision_score
import kornia

from dataset import MVTecDataset_Classification
from model_unetskip import ReconstructiveSubNetwork
from loss import SSIM
from pseudo_anomaly import Pseudo_Maker
import generate_noise as gn
from msgms import MSGMSLoss
from generic_util import trapezoid
from pro_curve_util import compute_pro


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)



def train_on_device(obj_names, args):
    #高速化、再現性なくなる(関係ない) 50s/epoch => 49s/epoch
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'FAIR'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=1)
        # Pytorch2.0からの高速化 49s/epoch => 38s/epoch
        #https://rest-term.com/archives/3708/
        # model = torch.compile(model)

        model.cuda()
        model.apply(weights_init)

        noise_gen = gn.Perlin_Noise_Generater(noise_scale_min=0, noise_scale_max=6, th = 0.5,
                 texture=[gn.through, gn.mesh, gn.noise, gn.cross_check]
                 )
        # noise_gen = gn.Simplex_Noise_Generater(noise_scale=[1, 7], noise_aspect=[0.2, 1.], th = 0.7,
        #          texture=[gn.through, gn.mesh, gn.noise, gn.cross_check]
        #          )
        Anomaly_Maker = Pseudo_Maker([256,256],
                                    model_path="/mnt/kamiya/code/VQGANVAE/OpenImages-8192/last.ckpt",
                                    config_path="/mnt/kamiya/code/VQGANVAE/OpenImages-8192/model.yaml").cuda()
        Anomaly_Maker = torch.compile(Anomaly_Maker)

        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      ],weight_decay=0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_l1 = torch.nn.modules.loss.L1Loss()
        loss_ssim = SSIM()

        
        # JSONファイルを読み込みモードで開く
        with open('anomaly_mask.json', 'r') as file:
            # JSONファイルを辞書型に変換
            background_configs = json.load(file)[obj_name]
        # dataset = MVTecTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])
        train_dataset = MVTecDataset_Classification(mvtec_root="/mnt/kamiya/dataset/MVtec_AD",
                                    dtd_root="/mnt/kamiya/dataset/DTD/images",
                                    category=obj_name,
                                    input_size=[256,256],
                                    noise_generator=noise_gen,
                                    use_mask=background_configs["use_mask"],
                                    bg_threshold=background_configs["bg_threshold"],
                                    bg_reverse=background_configs["bg_reverse"],
                                    is_train=True)

        test_dataset = MVTecDataset_Classification(mvtec_root="/mnt/kamiya/dataset/MVtec_AD",
                                    dtd_root="/mnt/kamiya/dataset/DTD/images",
                                    category=obj_name,
                                    input_size=[256,256],
                                    use_mask=False,
                                    is_train=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs,
                                shuffle=True,
                                num_workers=os.cpu_count(), #57s/epoch => 50s/epoch
                                pin_memory=True
                                )

        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True
                                )
        n_iter = 0
        best_loss = 999999.
        best_Score = 0.0
        for epoch in range(args.epochs):
            sum_loss = 0.
            model.train()
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True) as pbar:
                for i_batch, sample_batched in pbar:
                    images = sample_batched["image"].cuda(non_blocking=True)
                    anomaly_source_images = sample_batched["anomaly"].cuda(non_blocking=True)
                    anomaly_mask = sample_batched["mask"].cuda(non_blocking=True)

                    images, anomaly_images, anomaly_mask = Anomaly_Maker(images, anomaly_source_images, anomaly_mask)

                    anomaly_mask = anomaly_mask.cuda(non_blocking=True)
                    # torchvision.utils.save_image(torch.cat([images,anomaly_images]),
                    #                 f"images.png",
                    #                 normalize=True)
                    anomaly_mask = mean_smoothing(anomaly_mask, 5)
                    anomaly_mask_max = anomaly_mask.flatten(1).max(dim=-1)[0][:, None, None, None]
                    anomaly_mask_min = anomaly_mask.flatten(1).min(dim=-1)[0][:, None, None, None]
                    anomaly_mask = (anomaly_mask - anomaly_mask_min) / (anomaly_mask_max - anomaly_mask_min + 1e-6)

                    inputs = torch.cat([images, anomaly_images], dim=0)
                    targets = torch.cat([torch.zeros_like(anomaly_mask), anomaly_mask], dim=0)
                    outputs = model(inputs)
                    outputs = F.sigmoid(outputs)
                    loss = loss_l2(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()

                    pbar.set_postfix({"epoch": epoch, "iters": n_iter, "loss": loss.item()})

                    if args.visualize and n_iter % 200 == 0:
                        visualizer.plot_loss(loss, n_iter, loss_name='l2_loss')
                    if args.visualize and n_iter % 400 == 0:
                        visualizer.visualize_image_batch(images, n_iter, image_name='images')
                        visualizer.visualize_image_batch(anomaly_images, n_iter, image_name='anomaly_images')
                        visualizer.visualize_image_batch(outputs[len(outputs)//2:], n_iter, image_name='predict_anomaly')
                        visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')

                    sleep(0.5)

                    n_iter +=1

            scheduler.step()
            auroc_pixel, auroc, au_pro = test_per_epoch(model, Anomaly_Maker, test_dataloader)
            print("AUC Image:  " +str(auroc))
            print("AUC Pixel:  " +str(auroc_pixel))
            print("PRO:  " +str(au_pro))
            all_score = (auroc_pixel + auroc + au_pro)/3
            if all_score > best_Score:
                best_Score = all_score
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+"_bestscore.pckl"))

            if best_loss > sum_loss / len(train_dataloader):
                best_loss = sum_loss / len(train_dataloader)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+"_bestloss.pckl"))

@torch.no_grad()
def test_per_epoch(model, Anomaly_Maker, dataloader):
    model.eval()
    img_dim = 256

    #calculate pro
    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataloader.dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataloader.dataset)))
    mask_cnt = 0

    #calculate pro
    pro_gt=[]
    pro_out=[]
    anomaly_score_gt = []
    anomaly_score_prediction = []

    for i_batch, sample_batched in tqdm(enumerate(dataloader), total=len(dataloader), leave=True):
        gray_batch = sample_batched["image"].cuda()
        true_mask = sample_batched["mask"]
        anomaly_score_gt.append(torch.any(true_mask != 0).item() * 1.)
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        #VQGANで作成すると精度下がるらしい
        gray_batch, _ = Anomaly_Maker(gray_batch)
        # gray_rec = model(recon_image)
        
        outputs = model(gray_batch)
        outputs = F.sigmoid(outputs)

        out_mask_averaged = outputs
        out_mask_averaged = out_mask_averaged.detach().cpu().numpy()


        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)
        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_averaged.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

        #for pro
        truegt=true_mask_cv[:,:,0]
        outresult=out_mask_averaged[0,0,:,:]
        pro_gt.append(truegt)
        pro_out.append(outresult)

    all_fprs, all_pros = compute_pro(
        anomaly_maps=pro_out,
        ground_truth_maps=pro_gt,
        num_thresholds=5000)

    au_pro = trapezoid(all_fprs, all_pros, x_max=0.3)
    au_pro /= 0.3
    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    return auroc_pixel, auroc, au_pro

if __name__=="__main__":
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--obj_id', action='store', type=int, required=True)
        parser.add_argument('--bs', action='store', type=int, required=True)
        parser.add_argument('--lr', action='store', type=float, required=True)
        parser.add_argument('--epochs', action='store', type=int, required=True)
        parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
        parser.add_argument('--data_path', action='store', type=str, required=True)
        parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
        parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
        parser.add_argument('--log_path', action='store', type=str, required=True)
        parser.add_argument('--visualize', action='store_true')

        args = parser.parse_args()

        obj_batch = [['capsule'],
                     ['bottle'],
                     ['carpet'],
                     ['leather'],
                     ['pill'],
                     ['transistor'],
                     ['tile'],
                     ['cable'],
                     ['zipper'],
                     ['toothbrush'],
                     ['metal_nut'],
                     ['hazelnut'],
                     ['screw'],
                     ['grid'],
                     ['wood']
                     ]

        if int(args.obj_id) == -1:
            obj_list = [
                        'wood',
                        'bottle',
                        'carpet',
                        'pill',
                        'transistor',
                        'tile',
                        'cable',
                        'zipper',
                        'toothbrush',
                        'metal_nut',
                        'hazelnut',
                        'grid',
                        'capsule',
                        'leather'
                        ]
            picked_classes = obj_list
        else:
            picked_classes = obj_batch[int(args.obj_id)]

        with torch.cuda.device(args.gpu_id):
            train_on_device(picked_classes, args)


