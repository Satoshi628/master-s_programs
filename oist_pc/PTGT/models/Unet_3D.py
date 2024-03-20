#coding: utf-8
#----- Standard Library -----#
#None

#----- Public Package -----#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

#----- Module -----#
#None


class DoubleConv(nn.Module):
    #(convolution => [BN] => ReLU) * 2
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm3d(mid_channels),
            Mish(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm3d(out_channels),
            Mish()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))

#FReLU
class FReLU(nn.Module):
    def __init__(self, in_c, k=3, s=1, p=1):
        super().__init__()
        #Depthwise Convolution
        #https://ai-scholar.tech/articles/treatise/mixconv-ai-367
        self.FReLU = nn.Conv3d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c, bias=False)
        self.bn = nn.BatchNorm3d(in_c)

    def forward(self, x):
        tx = self.bn(self.FReLU(x))
        out = torch.max(x, tx)
        return out

# No assign
class UNet_3D(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32,
                noise_strength=0.4,
                bilinear=True,
                **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, channel)
        self.down1 = Down(channel, channel*2)
        self.down2 = Down(channel*2, channel*4)
        self.down3 = Down(channel*4, channel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(channel*8, channel*16 // factor)
        self.up1 = Up(channel*16, channel*8 // factor, bilinear)
        self.up2 = Up(channel*8, channel*4 // factor, bilinear)
        self.up3 = Up(channel*4, channel*2 // factor, bilinear)
        self.up4 = Up(channel*2, channel, bilinear)
        self.outc = nn.Conv3d(channel, n_classes, kernel_size=1, bias=True, padding_mode='reflect')
        
        # kernel_size = 4*sigma+0.5
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength

    def coordinater(self, out):
        """Function to detect the position of an object from a probability map

        Args:
            out (tensor[batch,1,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """       
        h = out.flatten(0, 2)
        h = (h >= self.noise_strength) * (h == self.maxpool(h)) * 1.0
        for _ in range(3):
            h = self.gaussian_filter(h)
            h = (h != 0) * (h == self.maxpool(h)) * 1.0
        
        coordinate = torch.nonzero(h)  #[detection number, 3(batch,y,x)]
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(h.size(0))]
        coordinate = torch.nn.utils.rnn.pad_sequence(coordinate, batch_first=True, padding_value=-1)
        #[batch*length,num,2(y,x)] => [batch*length,num,2(x,y)]
        coordinate = coordinate[:, :, [1, 0]]

        coordinate = coordinate.view(out.size(0), out.size(2), -1, 2)
        return coordinate #[batch,length,num,2(x,y)]

    def forward(self, x, coord=None):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = torch.sigmoid(out)

        if coord is None:
            coord = self.coordinater(out)

        return out, x, coord  # EP map, feature map, coordinate

    def get_feature(self, x, coord):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x  # feature map

    def domain_adaptation(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x # feature map

    def tracking_process(self, x, coord=None, add_F_dict=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if coord is None:
            out = self.outc(x)
            out = torch.sigmoid(out)
            coord = self.coordinater(out)

        return x, coord, add_F_dict


class UNet_3D_FA(nn.Module):
    def __init__(self, in_channels, n_classes, channel=32,
                feature_num=512,
                overlap_range=100.,
                noise_strength=0.5,
                bilinear=True,
                **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, channel)
        self.down1 = Down(channel, channel*2)
        self.down2 = Down(channel*2, channel*4)
        self.down3 = Down(channel*4, channel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(channel*8, channel*16 // factor)
        self.up1 = Up(channel*16, channel*8 // factor, bilinear)
        self.up2 = Up(channel*8, channel*4 // factor, bilinear)
        self.up3 = Up(channel*4, channel*2 // factor, bilinear)
        self.up4 = Up(channel*2, channel, bilinear)
        self.outc = nn.Conv3d(channel, n_classes, kernel_size=1, bias=True)

        self.feature_num = feature_num
        self.addtional_feature = nn.Parameter(torch.empty(feature_num, channel))
        nn.init.xavier_uniform_(self.addtional_feature)
        
        self.overlap_range = overlap_range

        # def detection function
        # kernel_size = 4*sigma+0.5
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength

    def random_feature_addtion(self, feature_map, coord, len_=0):
        """Random Feature Assignment Module

        Args:
            feature_map (tensor[batch,channel,length,H,W]): feature map
            coord (tensor[batch,length,num,2]): detection coordinate, PAD=-1
            len_ (int): assign time, default=0

        Returns:
            tensor[batch,channel,length,H,W]: assigned feature map
        """      

        coord = coord[:, len_]

        PAD_flag = coord[:, :, 0] < 0
        PAD_flag = PAD_flag[:, :, None] | PAD_flag[:, None, :] #[batch,num,num]

        coord_feature_idx = torch.randint(self.feature_num, coord.shape[:2],device=coord.device)

        for _ in range(20):
            distance = (coord[:, :, None] - coord[:, None])**2
            distance = distance.sum(dim=-1)
            distance = distance <= self.overlap_range
            distance[PAD_flag] = False

            overlap_flag = coord_feature_idx[:, :, None] == coord_feature_idx[:, None, :]

            overlap_flag = (overlap_flag & distance).sum(dim=-1)
            overlap_flag = overlap_flag > 1

            if overlap_flag.sum() == 0:
                break
            coord_feature_idx[overlap_flag] = torch.randint(self.feature_num, [overlap_flag.sum()],device=coord.device)

        PAD_flag = coord[:, :, 0] < 0
        add_feature = self.addtional_feature[coord_feature_idx[~PAD_flag]]

        batch_idx = torch.arange(coord.size(0)*coord.size(1),dtype=torch.long, device=coord.device).view(coord.shape[:2]) // coord.size(1)

        out = feature_map.clone()
        out[batch_idx[~PAD_flag], :,len_,
                    coord[:, :, 1][~PAD_flag],
                    coord[:, :, 0][~PAD_flag]] = add_feature
        
        return out, {'coord': coord, 'coord_feature_idx': coord_feature_idx}

    def feature_addtion(self, feature_map, coord, coord_feature_idx, len_=0):
        """Fixed Feature Assignment Module

        Args:
            feature_map (tensor[batch,channel,length,H,W]): feature map
            coord (tensor[batch,length,num,2]): detection coordinate, PAD=-1
            coord_feature_idx (tensor[batch,length,num]): Place index to assign
            len_ (int): assign time, default=0

        Returns:
            tensor[batch,channel,length,H,W]: assigned feature map
        """        

        PAD_flag = coord[:, :, 0] < 0
        add_feature = self.addtional_feature[coord_feature_idx[~PAD_flag]]

        batch_idx = torch.arange(coord.size(0) * coord.size(1), dtype=torch.long, device=coord.device).view(coord.shape[:2]) // coord.size(1)

        out = feature_map.clone()
        out[batch_idx[~PAD_flag], :,len_,
                    coord[:, :, 1][~PAD_flag],
                    coord[:, :, 0][~PAD_flag]] = add_feature
        
        return out

    def coordinater(self, out):
        """Function to detect the position of an object from a probability map

        Args:
            out (tensor[batch,1,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """     
        h = out.flatten(0, 2)
        h = (h >= self.noise_strength) * (h == self.maxpool(h)) * 1.0
        for _ in range(3):
            h = self.gaussian_filter(h)
            h = (h != 0) * (h == self.maxpool(h)) * 1.0
        
        coordinate = torch.nonzero(h)  #[detection number, 3(batch,y,x)]
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(h.size(0))]
        coordinate = torch.nn.utils.rnn.pad_sequence(coordinate, batch_first=True, padding_value=-1)
        #[batch*length,num,2(y,x)] => [batch*length,num,2(x,y)]
        coordinate = coordinate[:, :, [1, 0]]

        coordinate = coordinate.view(out.size(0), out.size(2), -1, 2)
        return coordinate #[batch,length,num,2(x,y)]

    def forward(self, x, coord=None):
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = torch.sigmoid(out)

        if coord is None:
            coord = self.coordinater(out)
        new_x1, _ = self.random_feature_addtion(x1, coord)
        new_x1, _ = self.random_feature_addtion(new_x1, coord, len_=-1)
        
        x2 = self.down1(new_x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, new_x1)

        return out, x, coord  # EP map, feature map, coordinate

    def get_feature(self, x, coord):
        x1 = self.inc(x)
        new_x1, _ = self.random_feature_addtion(x1, coord)
        new_x1, _ = self.random_feature_addtion(new_x1, coord, len_=-1)

        x2 = self.down1(new_x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, new_x1)

        return x  # feature map

    def domain_adaptation(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x # feature map

    def tracking_process(self, x, coord=None, add_F_dict=None):
        x1 = self.inc(x)
        if coord is None:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            out = self.outc(x)
            out = torch.sigmoid(out)

            coord = self.coordinater(out)

        if add_F_dict is None:
            new_x1, _ = self.random_feature_addtion(x1, coord)
        else:
            new_x1 = self.feature_addtion(x1, add_F_dict['coord'], add_F_dict['coord_feature_idx'])
        
        new_x1, add_F_dict = self.random_feature_addtion(new_x1, coord, len_=-1)
        
        x2 = self.down1(new_x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, new_x1)

        return x, coord, add_F_dict