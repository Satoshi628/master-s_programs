#coding: utf-8
#----- Standard Library -----#
import os
import random
import json

#----- External Library -----#
import numpy as np
import pydicom as pyd
from glob import glob
from tqdm import tqdm

#----- My Module -----#
#None
PVT_True = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,29,30,31,32,33,34,35]
PVT_False = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,36,37,38,39,40,41,42]
PVT_onset = {}

for pvtT in PVT_True:
    PVT_onset[pvtT] = 1

for pvtF in PVT_False:
    PVT_onset[pvtF] = 0

def make_annotation_json(root_dir):
    data_paths = sorted(glob(os.path.join(root_dir,"data","*")))

    image_id = 0
    image_list = []
    annotation_list = []

    for patient_path in data_paths:
        img_paths = sorted(glob(patient_path + "/*"))
        #dcm_test.size()=>[H,W]
        dcm_test = pyd.dcmread(img_paths[0]).pixel_array

        width = dcm_test.shape[1]
        height = dcm_test.shape[0]
        data_type = str(dcm_test.dtype)

        image_list.extend([
            {"file_name":path,
            "height":height,
            "width":width,
            "id":image_id + idx,
            "frame_id":idx,
            "patient_id":int(patient_path.split("/")[-1]),
            "data_type":data_type,
            "seq_length":len(img_paths),
            "first_frame_image_id":image_id
            }
            for idx,path in enumerate(img_paths)
        ])

        annotation_list.extend([
            {"image_id":image_id + idx,
            "onset":PVT_onset[int(patient_path.split("/")[-1])],
            "patient_id":int(patient_path.split("/")[-1])
            }
            for idx,path in enumerate(img_paths)
        ])

        image_id = len(image_list)


    json_dict ={
        "images":image_list,
        "annotations":annotation_list
    }
    with open(root_dir + "/annotation.json",mode="w") as f:
        json.dump(json_dict,f,indent=4)

if __name__ == '__main__':
    #root="/mnt/hdd1/kamiya/dataset/PVT_dataset"
    root="/mnt/hdd1/kamiya/dataset/partially_PVT_dataset"
    make_annotation_json(root)
    pass


"""
#coding: utf-8
#----- Standard Library -----#
import os
import random
import json

#----- External Library -----#
import numpy as np
import pydicom as pyd
from glob import glob
from tqdm import tqdm

#----- My Module -----#
#None
PVT_True = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,29,30,31,32,33,34,35]
PVT_False = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,36,37,38,39,40,41,42]
PVT_onset = {}

for pvtT in PVT_True:
    PVT_onset[pvtT] = 1

for pvtF in PVT_False:
    PVT_onset[pvtF] = 0

def make_annotation_json(root_dir):
    data_paths = sorted(glob(os.path.join(root_dir,"data","*")))

    image_id = 0
    image_list = []
    annotation_list = []

    for patient_path in data_paths:
        img_paths = sorted(glob(patient_path + "/*"))
        #dcm_test.size()=>[H,W]
        dcm_test = pyd.dcmread(img_paths[0])
        #print(dcm_test.Manufacturer)
        dicom_info_key = [item for item in dir(dcm_test) if item[0] != "_"]
        print(dcm_test.StudyTime)
        #print(getattr(dcm_test,'PatientSex'))
        print(dicom_info_key)
        input()


if __name__ == '__main__':
    root="/mnt/kamiya/dataset/PVT_dataset"
    make_annotation_json(root)
    pass
"""
