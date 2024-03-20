#coding: utf-8
#----- Standard Library -----#
import os
import random
import json

#----- External Library -----#
import glob
import pydicom as pyd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import tifffile as tif
from PIL import Image

#----- My Module -----#
#None

def make_image_info():
    root_dir = "/mnt/kamiya/dataset/PVT_dataset"

    train = list(range(1,43))
    with open(root_dir + "/annotation.json",mode="r") as f:
        data_json = json.load(f)


    #need .astype(np.int16) because exist uint16
    image_list = [{"parient_id": img_data["parient_id"], "image": pyd.dcmread(img_data['file_name']).pixel_array[None]}
                for img_data in data_json["images"]
                if img_data["parient_id"] in train]

    #.astype(np.int16)
    all_count = {idx+1:0 for idx in range(42)}
    int_count = {idx+1:0 for idx in range(42)}
    uint_count = {idx+1:0 for idx in range(42)}
    for img in image_list:
        if img["image"].dtype == np.int16:
            #print(img["image"].max())
            #print(img["image"].min())
            int_count[img["parient_id"]] += 1

        if img["image"].dtype == np.uint16:
            #print(img["image"].max())
            #print(img["image"].min())
            uint_count[img["parient_id"]] += 1
        all_count[img["parient_id"]] += 1

    print(all_count)
    print(int_count)
    print(uint_count)
    with open("/mnt/kamiya/dataset/PVT_dataset/image_info.csv","w") as f:
        f.write("parient_id\tall frames\tint frames\tuint frames\tdata type\n")
        for idx in range(1,43):
            text = ""
            if int_count[idx] != 0:
                text = text + "\tint16"
            if uint_count[idx] != 0:
                text = text + "\tuint16"
            f.write(f"{idx}\t{all_count[idx]}\t{int_count[idx]}\t{uint_count[idx]}{text}\n")


def make_hist():
    root_dir = "/mnt/kamiya/dataset/PVT_dataset"

    uint = [6,8,10,13,14,27,35]

    with open(root_dir + "/annotation.json",mode="r") as f:
        data_json = json.load(f)


    #need .astype(np.int16) because exist uint16
    image_list = [pyd.dcmread(img_data['file_name']).pixel_array
                for img_data in data_json["images"]
                if img_data["patient_id"] in uint]
    
    images = np.array(image_list).flatten()
    images = np.clip(images,-2048,2048)
    plt.clf()
    plt.hist(images, bins=100)
    plt.title("uint image histgram")
    plt.savefig("uint_hist.png")
    plt.close()

def image_type():
    #need .astype(np.int16) because exist uint16
    image = pyd.dcmread("/mnt/kamiya/dataset/PVT_dataset/data/0006/00000000").pixel_array.astype(np.int16)-1024
    print(image.shape)
    print(image.min())
    print(image.max())
    print(image.dtype)
    images = np.array(image).flatten()
    plt.clf()
    plt.hist(images, bins=100)
    plt.title("calibration image histgram")
    plt.savefig("test.png")
    plt.close()

def test():
    root_dir = "/mnt/kamiya/dataset/PVT_dataset"
    with open(root_dir + "/annotation.json",mode="r") as f:
        data_json = json.load(f)


    #need .astype(np.int16) because exist uint16
    image_list = [pyd.dcmread(img_data['file_name']).pixel_array
                for img_data in data_json["images"]
                if img_data["patient_id"] == 6]
    
    pass

def match():
    A_number = "0024"
    P_number = "0024no25"

    A_folder = f"/mnt/kamiya/dataset/PVT_dataset/data/{A_number}"
    P_folder = f"/mnt/kamiya/dataset/partially_PVT_dataset/{P_number}"
    A_files = glob.glob(A_folder+"/0*")
    P_files = glob.glob(P_folder+"/0*")

    A_images = torch.tensor(np.array([pyd.dcmread(file).pixel_array for file in A_files]))
    P_images = torch.tensor(np.array([pyd.dcmread(file).pixel_array for file in P_files]))
    A_images = A_images.flatten(-2)
    P_images = P_images.flatten(-2)
    match_map = torch.all(A_images[:,None] == P_images[None],dim=-1)
    print(match_map.shape)
    print(match_map.sum(dim=-1))
    print(torch.nozeros(match_map.sum(dim=-1)==0))

def new_data_read_test():
    folders = glob.glob("/mnt/kamiya/dataset/partially_PVT_dataset/data/*")
    files = [glob.glob(f+"/*") for f in folders]
    files = [f2 for f in files for f2 in f]
    n = 0
    for file in files:
        try:
            data = pyd.dcmread(file).pixel_array
            print(data.shape)
            print(data.dtype)
        except:
            n += 1
            print(file)

#new_data_read_test()
image_type()
