import os
import glob



dataset_path = "/mnt/kamiya/dataset/MVtec_AD-N2A"

image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "ground_truth", "pseudo_anomaly", "*")))

for path in image_paths:
    file_name = os.path.basename(path)
    file_name = "pseudo_" + file_name
    dirname = os.path.dirname(path)

    os.rename(path, os.path.join(dirname, file_name))