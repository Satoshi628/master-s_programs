import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 初期フィギュアと3D軸を設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# アニメーションのデータを生成
def gen_data(t):
    x = np.sin(t) + np.random.standard_normal(10) * 0.1
    y = np.cos(t) + np.random.standard_normal(10) * 0.1
    z = t + np.random.standard_normal(10) * 0.1
    return x, y, z

# アニメーションの更新関数を定義
def update(t):
    ax.clear()  # 軸をクリア
    x, y, z = gen_data(t)
    ax.scatter(x, y, z, s=3., linewidths=0.0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 10)

# アニメーションを作成
ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), blit=False)

# アニメーションを表示（Jupyter Notebook内での表示には追加設定が必要な場合があります）
# plt.show()

# アニメーションを保存（ffmpegがインストールされている必要があります）
ani.save('animation.mp4', writer='ffmpeg', fps=30)
