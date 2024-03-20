import os

import numpy as np
import matplotlib.pyplot as plt


category = "leather"
save_dir = f"results/MVTecAD_Results_Normal2/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0_0/models/mvtec_{category}-None"
pos_path = os.path.join(save_dir, "position.npz")

#データロード
pos = np.load(pos_path)["arr_0"]
print(pos.shape)

#2つとも重複を削除したい
#2つの値(x,y)を1つに圧縮
pos = pos[:, 0]*100 + pos[:, 1]
x100_y, counts_x100_y = np.unique(pos, return_counts=True)

counts_map = np.zeros([28,28])
for xy, c in zip(x100_y, counts_x100_y):
    counts_map[xy%100, xy//100] = c

fig, ax = plt.subplots()
img = ax.imshow(counts_map, cmap='jet')
fig.colorbar(img, ax=ax)
plt.savefig(f"_positon_{category}.png")