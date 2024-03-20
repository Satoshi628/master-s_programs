import os
from glob import glob
import numpy as np
import cv2

data_type = "datanew2"
topK = 100

data = np.loadtxt(f"{data_type}.csv", delimiter="\t", dtype=str)

image_name = data[:,0]
abs_res = data[:,1].astype(np.float32)
res = data[:,2].astype(np.float32)


#降順
abs_idx = np.argsort(abs_res)[::-1]
#昇順
non_abs_idx = np.argsort(res)

for i, idx in enumerate(abs_idx[:topK]):
    name = str(image_name[idx])
    img_path = f"{data_type}" + "\\" + name
    fastflow_img_path = f"FastFlow_{data_type}" + "\\" + name
    label_img_path = f"FastFlow_label_{data_type}" + "\\" + name
    PatchCore_img_path = f"PatchCore_{data_type}"+ f"\\images" + f"\\{name}"
    SimpleNet_img_path = f"SimpleNet_{data_type}" + "\\" + name

    img = cv2.imread(img_path)
    fastflow_img = cv2.imread(fastflow_img_path)
    label_img = cv2.imread(label_img_path)
    PatchCore_img = cv2.imread(PatchCore_img_path)
    SimpleNet_img = cv2.imread(SimpleNet_img_path)

    img = cv2.resize(img, fastflow_img.shape[:2][::-1])
    PatchCore_img = cv2.resize(PatchCore_img, fastflow_img.shape[:2][::-1])

    cat_img = np.concatenate([img, label_img, fastflow_img, PatchCore_img, SimpleNet_img], axis=1)
    cv2.imwrite(os.path.join(f"result_abs_{data_type}", f"{i:05}_{name}"), cat_img)


for i, idx in enumerate(non_abs_idx[:topK]):
    name = str(image_name[idx])

    img_path = f"{data_type}" + "\\" + name
    fastflow_img_path = f"FastFlow_{data_type}" + "\\" + name
    label_img_path = f"FastFlow_label_{data_type}" + "\\" + name
    PatchCore_img_path = f"PatchCore_{data_type}"+ f"\\images" + f"\\{name}"
    SimpleNet_img_path = f"SimpleNet_{data_type}" + "\\" + name

    img = cv2.imread(img_path)
    fastflow_img = cv2.imread(fastflow_img_path)
    label_img = cv2.imread(label_img_path)
    PatchCore_img = cv2.imread(PatchCore_img_path)
    SimpleNet_img = cv2.imread(SimpleNet_img_path)

    img = cv2.resize(img, fastflow_img.shape[:2][::-1])
    PatchCore_img = cv2.resize(PatchCore_img, fastflow_img.shape[:2][::-1])

    cat_img = np.concatenate([img, label_img, fastflow_img, PatchCore_img, SimpleNet_img], axis=1)
    cv2.imwrite(os.path.join(f"result_no_abs_{data_type}", f"{i:05}_{name}"), cat_img)