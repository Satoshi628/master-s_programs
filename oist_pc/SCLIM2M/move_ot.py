import os
import time
from tifffile import TiffFile
import numpy as np
import ot
import torch
import torch.nn.functional as F
import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

path = "/mnt/kamiya/dataset/SCLIM2M/02G-5000.tif"

def load_3DSCLIM2M(path):
    # TIFFファイルを読み込む
    with TiffFile(path) as tif:
        #なぜかこれを実行しないとすべてを取得できない
        print(tif.flags)

        images = tif.asarray()

    # 9894 = (51 * 194, 384, 384)
    images = images.reshape(51, -1, 384, 384).transpose(0, 2, 3, 1)
    return images #[51,384,384,194]

@torch.no_grad()
def calc_distance_matrix(vector1, vector2, batch_size=2048):
    """距離行列を計算するコード

    Args:
        vector (torch.tensor, gpu): [N,D]

    Returns:
        [torch.tensor, gpu]: [N,N]
    """

    distance_matrix = []
    distance_matrix_ap = distance_matrix.append
    for i in tqdm.tqdm(range(0, len(vector1), batch_size)):
        batch = vector1[i:i+batch_size]  # バッチを取得
        b_times_b = (batch ** 2).sum(-1)[:, None]
        v_times_v = (vector2 ** 2).sum(-1)[None]
        b_times_v = torch.matmul(batch, vector2.T)
        
        #[b,N]
        distance_matrix_ap((b_times_b -2*b_times_v + v_times_v).to("cpu"))

    distance_matrix = torch.cat(distance_matrix, 0)
    print(distance_matrix.shape)
    return distance_matrix.to("cuda:0")

def calc_transport(images):
    x = torch.linspace(0, 1, images.shape[-3])
    y = torch.linspace(0, 1, images.shape[-2])
    z = torch.linspace(0, 1, images.shape[-1])
    x, y, z = torch.meshgrid(x, y, z)
    xyz = torch.stack([x, y, z], axis=-1).to("cuda:0")


    if not os.path.exists("transport"):
        os.mkdir("transport")

    for idx in range(len(images)-1):
        pre_image = images[idx]
        nxt_image = images[idx+1]
        
        use_pre_flag = pre_image > 1.
        use_nxt_flag = nxt_image > 1.
        
        # use_pre_flag = pre_image > -1.
        # use_nxt_flag = nxt_image > -1.
        

        use_pre_xyz = xyz[use_pre_flag]
        use_pre_img = pre_image[use_pre_flag]
        use_pre_img = use_pre_img/torch.clamp(use_pre_img.sum(), min=1e-6)

        use_nxt_xyz = xyz[use_nxt_flag]
        use_nxt_img = nxt_image[use_nxt_flag]
        use_nxt_img = use_nxt_img/torch.clamp(use_nxt_img.sum(), min=1e-6)
        
        print(use_pre_xyz.shape)
        print(use_nxt_xyz.shape)
        distance_matrix = calc_distance_matrix(use_pre_xyz, use_nxt_xyz)
        ot_plan = ot.emd(use_pre_img, use_nxt_img, distance_matrix, numThreads="max")

        np.savez_compressed(f'transport/save_transport_{idx}', transport=ot_plan.to("cpu").numpy(), pre_xyz=use_pre_xyz.to("cpu").numpy(), nxt_xyz=use_nxt_xyz.to("cpu").numpy())


# 画像データを生成する関数
def generate_image_data(value):
    """
    0から1の値を受け取り、それに応じて色が変わる画像データを生成する。
    0は黒、1は緑色となる。
    """
    # 画像サイズを定義
    image_size = (value.shape[0], value.shape[1], 3)
    # 画像データのベースを生成（すべて黒）
    image_data = np.zeros(image_size)
    # 緑色の成分だけを値に応じて変更
    image_data[:, :, 1] = value
    return image_data


def show_move(images, path):
    images = images/images.max()
    images = images.cpu().numpy().mean(-1)
    # 動画を生成する
    fig = plt.figure()

    # アニメーションのフレームを生成する関数
    def update_frame(i):
        plt.clf()  # 前のフレームをクリア
        image_data = generate_image_data(images[int(i)])
        plt.imshow(image_data)
        plt.axis('off')  # 軸を非表示にする

    # アニメーションの作成
    ani = animation.FuncAnimation(fig, update_frame, frames=np.linspace(0, 50, 51), interval=50)

    # 動画を保存
    ani.save('green_gradient_animation.mp4', writer='ffmpeg', dpi=200, fps=4)


images = load_3DSCLIM2M(path)
# images = np.random.rand(51,384,384,194)
images = torch.from_numpy(images.astype(np.float32))
#down sample
sampling = 2
images = F.interpolate(images[None], scale_factor=1/sampling, mode="trilinear")[0]
# crop
#[x1,y1,z1,x2,y2,z2]
crop_bbox = [75,75,90,125,125,110]
images = images[:,crop_bbox[0]:crop_bbox[3],crop_bbox[1]:crop_bbox[4],crop_bbox[2]:crop_bbox[5]]
images = images.to("cuda:0")
print(images.shape)
calc_transport(images)

show_move(images, path)
