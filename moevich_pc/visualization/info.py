#coding: utf-8
#----- Standard Library -----#
import os
import random
import json

#----- External Library -----#
from tqdm import tqdm
import numpy as np
import pandas as pd
import pydicom as pyd
from glob import glob

#----- My Module -----#
#None

PVT_True = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,29,30,31,32,33,34,35]
PVT_False = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,36,37,38,39,40,41,42]
PVT_onset = {}

for pvtT in PVT_True:
    PVT_onset[pvtT] = True

for pvtF in PVT_False:
    PVT_onset[pvtF] = False

def my_get_attr(class_,memba):
    try:
        out = getattr(class_,memba)
    except:
        out = None
    return out


need_info = ["AcquisitionDate","AcquisitionType","BodyPartExamined",
            "ContrastBolusAgent","ContrastBolusStartTime","ContrastBolusVolume",
            "SeriesDate","InstitutionName","Manufacturer","ManufacturerModelName",
            "ImageType"]

def make_annotation_json(root_dir):
    data_paths = sorted(glob(os.path.join(root_dir,"/mnt/kamiya/dataset/PVT_dataset/data","*")))

    image_id = 0
    image_info_list = []
    for patient_path in data_paths:
        img_paths = sorted(glob(patient_path + "/*"))

        #dcm_test.size()=>[H,W]
        dcm_test = pyd.dcmread(img_paths[0])
        #print(dcm_test.Manufacturer)
        """
        dicom_info_key = [item for item in dir(dcm_test) if item[0] != "_"]
        for k in dicom_info_key:
            print(k)
        """
        info_dict = {
            "patient id":int(patient_path.split("/")[-1]),
            "PVT":PVT_onset[int(patient_path.split("/")[-1])],
            "data type":str(dcm_test.pixel_array.dtype),
            **{k:my_get_attr(dcm_test,k) for k in need_info}
        }

        image_info_list.append(info_dict)

    df = pd.DataFrame(image_info_list)
    if not os.path.exists("result"):
        os.mkdir("result")
    
    df.to_csv("result/data_info.csv")

if __name__ == '__main__':
    root="/mnt/kamiya/dataset/PVT_dataset"
    make_annotation_json(root)
    pass