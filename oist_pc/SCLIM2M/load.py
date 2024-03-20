from tifffile import TiffFile
import imageio
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

path = "/mnt/kamiya/dataset/SCLIM2M/02G-5000.tif"


# TIFFファイルを読み込む
with TiffFile(path) as tif:
    print(dir(tif))
    print(dir(tif.tiff))
    print(tif.flags)
    images = tif.asarray()

# imagesはnumpy配列として読み込まれるため、3Dデータや動画フレームとして扱うことができます。
# 例えば、最初のフレームを表示する場合:
# 9894 = (51 * 194, 384, 384)
images = images.reshape(51, -1, 384, 384).transpose(0, 2, 3, 1)

x = np.arange(0, images.shape[1])[:, None, None]
y = np.arange(0, images.shape[2])[None, :, None]
z = np.arange(0, images.shape[3])[None, None, :]
x = np.tile(x, (1, images.shape[2], images.shape[3])).flatten()
y = np.tile(y, (images.shape[1], 1, images.shape[3])).flatten()
z = np.tile(z, (images.shape[1], images.shape[2], 1)).flatten()

# アニメーションの更新関数を定義
def update(t):
    img = images[int(t)].flatten()
    flags = img > 5000*2
    img = img[flags]
    x_ten = x[flags]
    y_ten = y[flags]
    z_ten = z[flags]
    ax.clear()  # 軸をクリア
    ax.scatter(x_ten, y_ten, z_ten, c=img, cmap='Blues')
    # ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(-1.5, 1.5)
    # ax.set_zlim(0, 10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 散布図に靄の効果を追加
# アニメーションを作成
ani = FuncAnimation(fig, update, frames=np.linspace(0, 50, 51), blit=False)


# アニメーションを保存（ffmpegがインストールされている必要があります）
ani.save('animation.mp4', writer='ffmpeg', fps=4)