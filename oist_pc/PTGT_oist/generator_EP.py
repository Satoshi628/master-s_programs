#coding: utf-8
#----- Standard Library -----#
import os
import glob

#----- Public Package -----#
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import h5py
from PIL import Image
import torch.nn as nn

#----- Module -----#
# None


class distance_gaussian_filter(nn.Module):
    def __init__(self, device, size, sigma):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = 4 * sigma + 9
        # make coordinate map
        coord_x_map = torch.arange(size[0], device=device, dtype=torch.float).view(1, -1, 1, 1).repeat(1, 1, size[1], 1)
        coord_y_map = torch.arange(size[1], device=device, dtype=torch.float).view(1, 1, -1, 1).repeat(1, size[0], 1, 1)
        self.coord_map = torch.cat([coord_x_map, coord_y_map], dim=0)

    def forward(self, x):
        # get point coordinate
        coordinate = torch.nonzero(x.mean(dim=0)).view(1, 1, -1, 2).permute(3, 0, 1, 2)
        if coordinate.size(-1) == 0:
            return x.new_zeros(x.size())
        
        gaussian = torch.norm(self.coord_map - coordinate.float(), dim=0)
        gaussian = torch.exp(-0.5 * (gaussian / self.sigma).pow(2))
        
        value, _ = gaussian.max(dim=-1)
        img = value.unsqueeze(0)
        return img

def Generator():
    gaussian_filter = distance_gaussian_filter("cuda:0", (512, 512), 2)
    
    # [frame,id,x,y]
    f = h5py.File("./data/Cell_Point_Annotation.hdf5", mode='r')
    Density_paths = sorted([folder for folder in glob.glob(f"./data/*") if os.path.isdir(folder)])

    f_save = h5py.File("./data/Molecule_EP.hdf5", mode='a')

    for item_den_path in Density_paths:
        print("{}:\n".format(item_den_path.split("/")[-1]))
        data = f[item_den_path.split("/")[-1]][...]
        # make group
        try:
            group_save = f_save.create_group(item_den_path.split("/")[-1])
        except:
            group_save = f_save[item_den_path.split("/")[-1]]
        
        frame_len = len(glob.glob(item_den_path + "/video/*.png"))
        
        #swap x and y [x,y] => [y,x]
        track_let = torch.from_numpy(data[:, [0, 3, 2]])
        
        track_let = torch.round(track_let).to(torch.int32).transpose(0, 1)
        image_size = [512,512]
        sparse = torch.sparse_coo_tensor(track_let, torch.ones(track_let.shape[1]), (frame_len, image_size[0], image_size[1]))
        
        video = sparse.to_dense()
        video = video.unsqueeze(1)
        
        # save EP map
        for idx,img in enumerate(tqdm(video, leave=False)):
            img = img.cuda("cuda:0", non_blocking=True)
            
            img = gaussian_filter(img)

            img = img.to('cpu').detach().numpy().astype(np.float32)

            try:
                group_save.create_dataset(name=f'{idx:04}', data=img.astype('float32'), compression='gzip', compression_opts=9)
            except:
                del group_save[f'{idx:04}']
                group_save.create_dataset(name=f'{idx:04}', data=img.astype('float32'), compression='gzip', compression_opts=9)


Generator()

