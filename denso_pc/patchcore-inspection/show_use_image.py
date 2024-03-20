import os
import math

import numpy as np
import matplotlib.pyplot as plt


category = "bottle"
save_list = ["0", "6", "11", "16", "21", "26", "31", "36", "41", "46"]
for num in save_list[::-1]:
    save_dir = f"results/MVTecAD_Results_Few/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0_{num}"

    use_img_path = os.path.join(save_dir, "models", f"mvtec_{category}", "use_image_idx.npz")
    result_path = os.path.join(save_dir, "results.csv")
    data_rate = float(np.loadtxt(result_path, delimiter=",", dtype=str)[1,1])

    #データロード
    img_idx = np.load(use_img_path)["arr_0"]
    bins = round(1 + math.log2(img_idx.shape[0]))
    print(bins)
    plt.hist(img_idx, bins=bins, alpha=0.7, label=f"{data_rate}")
    plt.legend()

    plt.savefig(f"images/_use_img_{category}_{data_rate}.png")