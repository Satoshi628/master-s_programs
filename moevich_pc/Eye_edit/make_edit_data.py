#coding: utf-8
#----- Standard Library -----#
import os
import random
import json
import re
import shutil

#----- External Library -----#
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

#----- My Module -----#
# None
root_dir = "/mnt/hdd1/kamiya/dataset/Eye"

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_dataset():
        info = pd.read_csv(os.path.join(root_dir, "info_smoking.csv"))
        AMD_info = pd.read_csv(os.path.join(root_dir, "AMD-AI対応表 450iAMD.csv"))
        
        image_paths = sorted(glob.glob(os.path.join(root_dir,"images_edit","*")))
        OCT_paths = sorted(glob.glob(os.path.join(root_dir,"AI-AMD-OCT200","*.tif")))
        OCT_paths2 = sorted(glob.glob(os.path.join(root_dir,"AI-AMD-OCT400","*.tif")))

        image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
        image_num = np.array(image_num)

        OCT_num = [int(re.search(r"/OCT\d+",path).group()[4:]) for path in OCT_paths]
        OCT_num = np.array(OCT_num)

        OCT_num2 = [int(re.search(r"/OCT\d+",path).group()[4:]) for path in OCT_paths2]
        OCT_num2 = np.array(OCT_num2)
        
        label = [info.query(f"AI_number == {num}").iloc[:,[2,3,4,5,6]].values for num in image_num]
        label = np.concatenate(label, axis=0).squeeze()
        amd = [AMD_info.query(f"AI_number == {num}").iloc[:,[-1]].values for num in image_num]
        amd = np.concatenate(amd, axis=0)[:,None,0]
        amd = amd[:label.shape[0]-amd.shape[0]]
        label = np.concatenate([label, amd[:,0,None]], axis=-1)

        # bainary => Decimal
        #label = label[:,0,0]

        split_idx = np.arange(len(image_paths))

        images = [cv2.cvtColor(cv2.imread(image_paths[idx]),cv2.COLOR_BGR2RGB).transpose(2,0,1) for idx in split_idx]
        image_paths = [image_paths[idx] for idx in split_idx]
        
        OCT = [[] for _ in split_idx]
        OCT_paths = [[] for _ in split_idx]
        for idx in split_idx:
            if len(OCT_paths) > idx:
                OCT_idx = np.nonzero(OCT_num == idx)[0]
                for OCT_i in OCT_idx:
                    try:
                        OCT[idx].append(cv2.cvtColor(cv2.imread(OCT_paths[OCT_i]), cv2.COLOR_BGR2RGB).transpose(2,0,1))
                        OCT_paths[idx].append(OCT_paths[OCT_i])
                    except:
                        OCT[idx].append(np.zeros([3, 945, 945]))
                        OCT_paths[idx].append(None)

            else:
                OCT[idx].append(np.zeros([3, 945, 945]))
                OCT_paths[idx].append(None)
            if len(OCT_paths2) > idx:
                OCT_idx2 = np.nonzero(OCT_num2 == idx)[0]
                for OCT_i in OCT_idx2:
                    try:
                        OCT[idx].append(cv2.cvtColor(cv2.imread(OCT_paths2[OCT_i]), cv2.COLOR_BGR2RGB).transpose(2,0,1))
                        OCT_paths[idx].append(OCT_paths2[OCT_i])
                    except:
                        OCT[idx].append(np.zeros([3, 945, 945]))
                        OCT_paths[idx].append(None)
            else:
                OCT[idx].append(np.zeros([3, 945, 945]))
                OCT_paths[idx].append(None)

        labels = np.concatenate([label[idx]  for idx in split_idx], axis=0)

        make_dir("images")
        make_dir("OCT")

        for idx, (img_path, OCT_path) in enumerate(zip(image_paths, OCT_paths)):
            shutil.copyfile(img_path, os.path.join("images", f"{idx:04}.TIF"))

            if not OCT_path:
                cv2.imwrite(os.path.join("OCT", f"{idx:04}-None.tif"), np.zeros([945, 945, 3],dtype=np.uint8))

            for i, p in enumerate(OCT_path):
                if p is not None:
                    shutil.copyfile(p, os.path.join("OCT", f"{idx:04}-{i}.tif"))
                else:
                    cv2.imwrite(os.path.join("OCT", f"{idx:04}-None.tif"), np.zeros([945, 945, 3],dtype=np.uint8))

        start_idx = idx

        info = pd.read_csv(os.path.join(root_dir, "CSC-AI_smoking.csv"))
        #CSC_labels = info.iloc[:,[0,1,2,5,6]].values
        #CSC,ID,AGE,GENDER,ARMS2,CFHI62V,備考,smoking
        #=> CSC, AGE, GENDER, ARSM2, CFH, smoking
        CSC_labels = info.iloc[:,[0,2,3,4,5,7]].values


        CSC_paths = sorted(glob.glob(os.path.join(root_dir,"CSC-data","*")))
        CSC_OCT_paths = sorted(glob.glob(os.path.join(root_dir,"CSC-OCT","*")))
        CSC_num = np.array([int(re.search(r"\d+.tif",path).group()[:-4]) for path in CSC_paths])
        CSC_OCT_num = np.array([int(re.search(r"\d+.tif",path).group()[:-4]) for path in CSC_OCT_paths])
        CSC_paths = np.array(CSC_paths)
        CSC_OCT_paths = np.array(CSC_OCT_paths)



        count = 0
        for idx in range(len(CSC_labels)):
            file_num = CSC_labels[idx, 0]
            CSC_p = CSC_paths[CSC_num == file_num]
            CSC_OCT_p = CSC_OCT_paths[CSC_OCT_num == file_num]
            if CSC_p.shape[0] != 0:
                CSC_p = CSC_p[0]
            else:
                CSC_p = None
            if CSC_OCT_p.shape[0] != 0:
                CSC_OCT_p = CSC_OCT_p[0]
            else:
                CSC_OCT_p = None
            
            if CSC_OCT_p is None and CSC_p is None:
                print("ないデータ:",file_num)
                continue
            start_idx += 1
            count += 1
            
            if CSC_p is not None:
                shutil.copyfile(CSC_p, os.path.join("images", f"{start_idx:04}.tif"))
            else:
                cv2.imwrite(os.path.join("images", f"{start_idx:04}.tif"), np.zeros([945, 945, 3],dtype=np.uint8))
            
            if CSC_OCT_p is not None:
                shutil.copyfile(CSC_OCT_p, os.path.join("OCT", f"{start_idx:04}.tif"))
            else:
                cv2.imwrite(os.path.join("OCT", f"{start_idx:04}.tif"), np.zeros([945, 945, 3],dtype=np.uint8))
        
        CSC_labels = np.concatenate([CSC_labels, np.zeros([CSC_labels.shape[0], 1])], axis=-1)
        print(CSC_labels[:,1:].shape)
        print(label.shape)
        label = np.concatenate([label, CSC_labels[:,1:]],axis=0)
        label[np.isnan(label)] = -1
        #AGE, GENDER, ARSM2, CFH, smoking, AMD
        np.savetxt("label.txt", label, fmt=["%d", "%d", "%d", "%d", "%d", "%d"], delimiter=",")


def shijo():
    #PID,AI number,age,sex,ARMS2A69S,CFHI62V, ARMS2 risk homo,CFH risk homo,iAMD,枚数
    info = pd.read_csv(os.path.join(root_dir, "20230711 眼科　四條.csv"))
    image_paths = sorted(glob.glob(os.path.join(root_dir,"20230711 眼科　四條","*.tif")))
    
    image_num = [int(path.split("/")[-1][:-6]) for path in image_paths]
    image_num = np.array(image_num)
    label = [info.query(f"AI_number == {num}").iloc[:,[2,3,4,5,8]].values for num in image_num]
    label = np.concatenate(label, axis=0)
    label = np.concatenate([label[:,:4], np.full([label.shape[0], 1], -1), label[:,-1:]], axis=-1)

    past_label = np.loadtxt("label.txt", delimiter=",")
    label = np.concatenate([past_label, label], axis=0)
    np.savetxt("label.txt", label, fmt=["%d", "%d", "%d", "%d", "%d", "%d"], delimiter=",")

    past_num = int(sorted(glob.glob(os.path.join("images","*")))[-1].split("/")[-1][:-4])

    past_num = past_num + 1
    for idx, img_path in enumerate(image_paths):
        shutil.copyfile(img_path, os.path.join("images", f"{idx+past_num:04}.TIF"))

        cv2.imwrite(os.path.join("OCT", f"{idx+past_num:04}-None.tif"), np.zeros([945, 945, 3],dtype=np.uint8))


if __name__ =="__main__":
    # shijo()
    # make_dataset()
    pass