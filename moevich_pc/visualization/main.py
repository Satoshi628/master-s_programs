#coding: utf-8
#----- Standard Library -----#
import os
import random
import json

#----- External Library -----#
from tqdm import tqdm
import pydicom as pyd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Resize
import matplotlib.pyplot as plt
import matplotlib.animation as ani

#----- My Module -----#
from utils.utils import CT_ndarrayToTensor
from utils.dataset import PVT_Loader

############## dataloader関数##############
def dataload():
    ### data augmentation + preprocceing ###
    transform = Compose([CT_ndarrayToTensor(WW=0, WL=0),  # [0,1] + Tensor型に変換
                        ])
    
    root_dir = "/mnt/kamiya/dataset/PVT_dataset"
    dataset = PVT_Loader(root_dir=root_dir, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=2)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return dataloader

def CT_images_save(images,name):
    plt.clf()
    fig = plt.figure()
    #no memory
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    #video must be 2d list
    video =[[plt.imshow(img)] for img in images]
    
    video = ani.ArtistAnimation(fig, video,interval=125)
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists("result/CT_images"):
        os.mkdir("result/CT_images")
    #保存
    video.save(f"result/CT_images/patient_{name:03}.gif",
                writer="pillow")
    plt.close()

def histgram_save(name):
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists("result/hist"):
        os.mkdir("result/hist")
    
    root_dir = "/mnt/kamiya/dataset/PVT_dataset"

    with open(root_dir + "/annotation.json",mode="r") as f:
        data_json = json.load(f)


    #need .astype(np.int16) because exist uint16
    image_list = [pyd.dcmread(img_data['file_name']).pixel_array
                for img_data in data_json["images"]
                if img_data["patient_id"] == name]
    
    data_type = image_list[0].dtype    
    
    images = np.array(image_list).flatten()
    images = np.clip(images,-2048,2048)
    
    plt.clf()
    plt.hist(images, bins=100)
    plt.title(f"Data Type:{str(data_type)}")
    plt.savefig(f"result/hist/patient_{name:03}_{str(data_type)}.png")
    plt.close()
    pass

def main():
    dataloader = dataload()
    patient_images = {i:[] for i in range(1,43)}
    for image, info in tqdm(dataloader, leave=False):
        #image save
        #img = (image*255)[0].permute(1,2,0).repeat(1,1,3).numpy().astype(np.uint8)
        
        #histgram
        img = image.numpy()
        img = np.clip(img,-2048,2048)

        patient_images[info['patient_id'].item()].append(img)
    
    for k,v in patient_images.items():
        print(k)
        #CT_images_save(v,k)
        histgram_save(k)
    


############## main ##############
if __name__ == '__main__':
    main()