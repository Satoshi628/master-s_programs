import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import gaussian, create_window


def savefig(img,outmask,truemask,path,reconimg):
    grid = ImageGrid(
        fig=plt.figure(figsize=(16, 4)),
        rect=111,
        nrows_ncols=(1, 4),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )
    img=img.permute(0,2,3,1)
    img = img[0, :, :, :]
    img = img.cpu().numpy()
    img = img* 220
    img = img.astype('uint8')
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    reconimg=reconimg.permute(0,2,3,1)
    reconimg = reconimg[0, :, :, :]
    reconimg = reconimg.cpu().numpy()
    reconimg = reconimg* 220
    reconimg = reconimg.astype('uint8')
    # reconimg=cv2.cvtColor(reconimg,cv2.COLOR_BGR2RGB)

    grid[0].imshow(img)
    grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[0].set_title("Input Image", fontsize=14)

    grid[1].imshow(reconimg)
    grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[1].set_title("Recon Image", fontsize=14)

    grid[2].imshow(truemask)
    grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[2].set_title("GroundTruth", fontsize=14)

    grid[3].imshow(img)
    im = grid[3].imshow(outmask, alpha=0.3, cmap="jet")
    grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[3].cax.colorbar(im)
    grid[3].cax.toggle_label(True)
    grid[3].set_title("Anomaly Map", fontsize=14)

    plt.savefig(path, bbox_inches="tight")
    plt.close()




def ssim(img1, img2, window, window_size=11):
    #img [0, 1]
    # img1 = torch.clamp(img1, 0., 1.)
    # img2 = torch.clamp(img2, 0., 1.)

    padd = window_size//2
    (_, channel, height, width) = img1.size()

    u1 = F.conv2d(img1, window, padding=padd, groups=channel)
    u2 = F.conv2d(img2, window, padding=padd, groups=channel)


    v1 = F.conv2d(img1 ** 2, window, padding=padd, groups=channel) - u1**2
    v2 = F.conv2d(img2 ** 2, window, padding=padd, groups=channel) - u2**2
    v12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - (u1 * u2)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    c3 = c2/2

    #輝度値の変化
    l = (2. * u1*u2 + c1) / (u1**2 + u2**2 + c1)
    #輝度値の分散
    c = (2. * torch.sqrt(torch.clamp(v1 * v2, 0.0)) + c2) / (v1 + v2 + c2)
    #構造の変化
    s = (2. * v12 + c3) / (torch.sqrt(torch.clamp(v1 * v2, 0.0)) + c3)
    
    ssim = l*c*s
    l = l.mean(dim=1)
    c = c.mean(dim=1)
    s = s.mean(dim=1)
    ssim = ssim.mean(dim=1)

    return l, c, s, ssim


class SSIM_Score(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.window = nn.Parameter(create_window(window_size, 3))

    def forward(self, imgs, recon_imgs):
        l, c, s, ssim_value = ssim(imgs, recon_imgs, self.window, self.window_size)
        return 1-l, 1-c, 1-s, 1-ssim_value


