import os
import math

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import kornia
import lpips

from dataset import MVTecDataset
from model_unetskip import ReconstructiveSubNetwork
from utils import savefig, SSIM_Score
from msgms import MSGMSLoss
from pseudo_anomaly import Pseudo_Maker
import generate_noise as gn



def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)

def save_fig(score, masks, path):
    score_no = score[masks == 0]
    score_ano = score[masks != 0]

    log_scale = score_ano.shape[0] > 1000
    
    bins = round(1 + math.log2(score_no.shape[0]))
    plt.hist(score_no, bins=bins, log=log_scale, color="blue", label="normal",alpha=0.3)

    bins = round(1 + math.log2(score_ano.shape[0]))
    plt.hist(score_ano, bins=bins, log=log_scale, color="red", label="anomaly",alpha=0.3)
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def test(obj_names, mvtec_path, checkpoint_path, base_model_name,saveimages):
    obj_auroc_pixel_list = []
    obj_pro_list= []
    obj_auroc_image_list = []
    

    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'
        if not os.path.exists(f"results/{obj_name}/image-level"):
            os.makedirs(f"results/{obj_name}/image-level")

        if not os.path.exists(f"results/{obj_name}/pixel-level"):
            os.makedirs(f"results/{obj_name}/pixel-level")
        
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()
        
        noise_gen = gn.Simplex_Noise_Generater(noise_scale=[1, 7], noise_aspect=[0.3, 1/0.3], th = 0.7,
                 texture=[gn.through, gn.mesh, gn.noise, gn.cross_check])
        Anomaly_Maker = Pseudo_Maker([256,256],
                                    noise_generator=noise_gen,
                                    # model_path="/mnt/kamiya/code/VQGANVAE/OpenImages-8192/last.ckpt",
                                    # config_path="/mnt/kamiya/code/VQGANVAE/OpenImages-8192/model.yaml"
                                    ).cuda()
        Anomaly_Maker = torch.compile(Anomaly_Maker)

        # dataset = MVTecTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataset = MVTecDataset(mvtec_root="/mnt/kamiya/dataset/MVtec_AD",
                                    dtd_root="/mnt/kamiya/dataset/DTD/images",
                                    category=obj_name,
                                    input_size=[256,256],
                                    is_train=False)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False,
                                num_workers=6
                                )

        ssim = SSIM_Score(11).cuda()
        msgms = MSGMSLoss().cuda()
        lp_loss = lpips.LPIPS(net='vgg', spatial=True).cuda() # closer to "traditional" perceptual loss, when used for optimization
        

        #calculate pro
        masks = []
        l_score = []
        c_score = []
        s_score = []
        ssim_score = []
        color_score = []
        msgms_score = []
        fair_score = []
        lpips_score = []
        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(dataloader), total=len(dataloader), leave=True):
                gray_batch = sample_batched["image"].cuda()
                true_mask = sample_batched["mask"]
                masks.append(true_mask.numpy())
                #VQGANで作成すると精度下がるらしい
                gray_batch, recon_image = Anomaly_Maker(gray_batch)
                # gray_rec = model(recon_image)
                gray_rec = model(gray_batch)
                l, c, s, ssim_value = ssim(gray_batch, gray_rec)
                rec_lab = kornia.color.rgb_to_lab(gray_rec)
                ori_lab = kornia.color.rgb_to_lab(gray_batch)
                colordif = (ori_lab-rec_lab)*(ori_lab-rec_lab)
                colorresult = colordif[:,1,:,:]+colordif[:,2,:,:]

                out_map = msgms(gray_rec, gray_batch, as_loss=False)
                lpips_value = lp_loss(2*(gray_rec-0.5), 2*(gray_batch-0.5))
                lpips_score.append(mean_smoothing(lpips_value, 21).cpu().numpy())
                # lpips_score.append(lpips_value.cpu().numpy())

                
                fair = out_map + colorresult[:, None, :, :]*0.0003
                fair = mean_smoothing(fair, 21)[:, 0]

                color_score.append(mean_smoothing(colorresult[:,None], 21).cpu().numpy())
                msgms_score.append(mean_smoothing(out_map, 21).cpu().numpy())
                # color_score.append(colorresult.cpu().numpy())
                # msgms_score.append(out_map.cpu().numpy())
                fair_score.append(fair.cpu().numpy())

                l_score.append(mean_smoothing(l[:, None], 21).cpu().numpy())
                c_score.append(mean_smoothing(c[:, None], 21).cpu().numpy())
                s_score.append(mean_smoothing(s[:, None], 21).cpu().numpy())
                ssim_score.append(mean_smoothing(ssim_value[:,None], 21).cpu().numpy())
                
                # l_score.append(l.cpu().numpy())
                # c_score.append(c.cpu().numpy())
                # s_score.append(s.cpu().numpy())
                # ssim_score.append(ssim_value.cpu().numpy())
        
        color_score = np.concatenate(color_score)
        msgms_score = np.concatenate(msgms_score)
        fair_score = np.concatenate(fair_score)
        l_score = np.concatenate(l_score)
        c_score = np.concatenate(c_score)
        s_score = np.concatenate(s_score)
        ssim_score = np.concatenate(ssim_score)
        lpips_score = np.concatenate(lpips_score)
        masks = np.concatenate(masks)
        
        save_fig(color_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/color_score.png")
        save_fig(msgms_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/msgms_score.png")
        save_fig(fair_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/fair_score.png")
        save_fig(l_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/l_score.png")
        save_fig(c_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/c_score.png")
        save_fig(s_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/s_score.png")
        save_fig(ssim_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/ssim_score.png")
        save_fig(lpips_score.flatten(), masks.flatten(), f"results/{obj_name}/pixel-level/lpips_score.png")

        color_score = color_score.reshape([color_score.shape[0], -1]).max(axis=-1)
        msgms_score = msgms_score.reshape([msgms_score.shape[0], -1]).max(axis=-1)
        fair_score = fair_score.reshape([fair_score.shape[0], -1]).max(axis=-1)
        l_score = l_score.reshape([l_score.shape[0], -1]).max(axis=-1)
        c_score = c_score.reshape([c_score.shape[0], -1]).max(axis=-1)
        s_score = s_score.reshape([s_score.shape[0], -1]).max(axis=-1)
        ssim_score = ssim_score.reshape([ssim_score.shape[0], -1]).max(axis=-1)
        lpips_score = lpips_score.reshape([lpips_score.shape[0], -1]).max(axis=-1)
        masks = masks.reshape([masks.shape[0], -1]).max(axis=-1)
        
        save_fig(color_score, masks, f"results/{obj_name}/image-level/color_score.png")
        save_fig(msgms_score, masks, f"results/{obj_name}/image-level/msgms_score.png")
        save_fig(fair_score, masks, f"results/{obj_name}/image-level/fair_score.png")
        save_fig(l_score, masks, f"results/{obj_name}/image-level/l_score.png")
        save_fig(c_score, masks, f"results/{obj_name}/image-level/c_score.png")
        save_fig(s_score, masks, f"results/{obj_name}/image-level/s_score.png")
        save_fig(ssim_score, masks, f"results/{obj_name}/image-level/ssim_score.png")
        save_fig(lpips_score, masks, f"results/{obj_name}/image-level/lpips_score.png")


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--saveimages', action='store_true')

    args = parser.parse_args()
    savepath = args.checkpoint_path
    #MVTec AD
    obj_list = [
                'carpet',
                'grid',
                'leather',
                'tile',
                'wood',
                'bottle',
                'cable',
                'capsule',
                'hazelnut',
                'metal_nut',
                'pill',
                'screw',
                'toothbrush',
                'transistor',
                'zipper',
                ]
    #VisA
    '''
    obj_list = ['capsules',
                'candle',
                'cashew',
                'chewinggum',
                'fryum',
                'macaroni2',
                'macaroni1',
                'pcb1',
                'pcb2',
                'pcb3',
                'pcb4',
                'pipe_fryum',
                ]
    '''
    with torch.cuda.device(args.gpu_id):
        # test(["transistor"], args.data_path, args.checkpoint_path, args.base_model_name,args.saveimages)
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name,args.saveimages)
