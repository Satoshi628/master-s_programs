import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


_CLASSNAMES = [
    "bottle", # 0
    "cable", # 1
    "capsule", # 2
    "carpet", # 3
    "grid", # 4
    "hazelnut", # 5
    "leather", # 6
    "metal_nut", # 7
    "pill", # 8
    "screw", # 9
    "tile", # 10
    "toothbrush", # 11
    "transistor", # 12
    "wood", # 13
    "zipper", # 14
]

embed_path = os.path.join("umap", f"{_CLASSNAMES[0]}_embedded_vector.npz")
patch_path = os.path.join("umap", f"{_CLASSNAMES[0]}_patch.npz")
#データロード
xy = np.load(embed_path)["arr_0"]
patch = np.load(patch_path)["arr_0"]
#[N,H,W,3]に変換
patch = patch.transpose(0, 2, 3, 1)
H, W = patch[0].shape[:2]

fig, ax = plt.subplots()

#画像のダミーを定義
imagebox = OffsetImage(patch[0], zoom=8.)
imagebox.image.axes = ax

sc = plt.scatter(xy[:, 0], xy[:, 1], s=1)


annot = AnnotationBbox(imagebox, xy=(0,0), xybox=(W,H),
                    xycoords="data", boxcoords="offset points", pad=0.5,
                    arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
annot.set_visible(False)
ax.add_artist(annot)

def update_annot(ind):
    i = ind["ind"][0]
    pos = sc.get_offsets()[i]
    annot.xy = (pos[0], pos[1])
    imagebox.set_data(patch[i])

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()