import os
import shutil

import glob
import numpy as np
from scipy.sparse.csgraph import connected_components
import umap
import cv2
import faiss
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def load_faiss_vector(filename):
    index = faiss.read_index(filename)
    print("faiss特徴再構成中")
    vector = index.reconstruct_n(0, index.ntotal)
    print("Umap変換中")
    embedding = umap.UMAP().fit_transform(vector)
    return embedding


save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_N2A_Normal/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models"
data_dirs = glob.glob(os.path.join(save_dir, "*"))
for data_path in data_dirs:
    faiss_path = os.path.join(data_path, "nnscorer_search_index.faiss")
    patch_path = os.path.join(data_path, "patch_images.npz")

    #データロード
    xy = load_faiss_vector(faiss_path)

    save_path = "/" + os.path.join(*data_path.split("/")[:-2], "umap")

    cate_name = data_path.split("/")[-1].replace("mvtec_", "").replace("-", "")

    mkdir(save_path)
    
    np.savez_compressed(os.path.join(save_path, f"{cate_name}_embedded_vector"), xy)
    shutil.copy(patch_path, os.path.join(save_path, f"{cate_name}_patch.npz"))