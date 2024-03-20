import os

import numpy as np
from scipy.sparse.csgraph import connected_components
import umap
import faiss
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def load_faiss_vector(filename):
    index = faiss.read_index(filename)
    print("faiss特徴再構成中")
    vector = index.reconstruct_n(0, index.ntotal)
    print("Umap変換中")
    embedding = umap.UMAP().fit_transform(vector)
    return embedding


save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_A2N/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/mvtec_bottle-broken_large"
faiss_path = os.path.join(save_dir, "nnscorer_search_index.faiss")
patch_path = os.path.join(save_dir, "patch_images.npz")

#データロード
xy = load_faiss_vector(faiss_path)
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