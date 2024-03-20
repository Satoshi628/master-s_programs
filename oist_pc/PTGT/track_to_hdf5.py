#coding: utf-8
#----- Standard Library -----#
import sys
import os
import glob
import re

#----- Public Package -----#
import numpy as np
import h5py

#----- Module -----#
#None


def sort_np(data):
    sort_index = np.argsort(data[:, 0])
    data = data[sort_index]
    return data

def dir_in_path(path):
    files = os.listdir(path)
    files_dir = [os.path.join("./data", f) for f in files if os.path.isdir(os.path.join(path, f))]
    return files_dir

OIST_datatase_dir = sorted(dir_in_path("./data"))
for path_1 in OIST_datatase_dir:
    print(path_1)
    # make save file
    hdf5_f = h5py.File(os.path.join("./data","Cell_Point_Annotation.hdf5"), mode='a')
    
    dat_paths = sorted(glob.glob(os.path.join(path_1, "Tracks", "*.dat")))

    data_list = np.empty([0, 4])

    for idx, path_2 in enumerate(dat_paths):
        data = np.loadtxt(path_2, dtype=str)
        
        if data.shape[0] == 0:
            print(path_2)
            continue
        
        if data.ndim == 2:
            data = data[:, :3].astype(np.float32)
        elif data.ndim == 1:
            data = data[:3].astype(np.float32).reshape(1, 3)

        # first id=1
        data = np.concatenate([np.full([data.shape[0], 1], idx + 1), data], axis=1)
        data_list = np.concatenate([data_list, data], axis=0)
    
    # Swap frame and id
    temp = data_list[:, 0].copy()
    data_list[:, 0] = data_list[:, 1]
    data_list[:, 1] = temp
    
    # sort at frame
    data_list = sort_np(data_list)

    # save hdf5 file
    try:
        dataset = hdf5_f.create_dataset(name=path_1.split("/")[-1], data=data_list, dtype=np.float32)
    except:
        del hdf5_f[path_1.split("/")[-1]]
        dataset = hdf5_f.create_dataset(name=path_1.split("/")[-1], data=data_list, dtype=np.float32)
    

