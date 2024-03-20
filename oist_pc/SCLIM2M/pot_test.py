import ot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 2つの離散分布
N = 100
x = np.linspace(-1.,1.,N)
y = np.linspace(-1.,1.,N)
x = np.tile(x[:,None], (1, N))
y = np.tile(y[None], (N, 1))
xy = np.stack([x,y],axis=0)

normal1_mean = np.array([0., 0.])[:,None,None]
normal1_std = np.array([0.01, 0.01])[:,None,None]

normal2_mean = np.array([0.5, 0.5])[:,None,None]
normal2_std = np.array([0.01, 0.01])[:,None,None]

normal3_mean = np.array([0.2, 0.2])[:,None,None]
normal3_std = np.array([0.01, 0.01])[:,None,None]

normal4_mean = np.array([0.7, 0.7])[:,None,None]
normal4_std = np.array([0.01, 0.01])[:,None,None]

normal1 = np.exp(-np.linalg.norm(xy-normal1_mean, axis=0)**2/np.sqrt(2* np.pi*np.prod(normal1_std)))
normal2 = np.exp(-np.linalg.norm(xy-normal2_mean, axis=0)**2/np.sqrt(2* np.pi*np.prod(normal2_std)))
normal3 = np.exp(-np.linalg.norm(xy-normal3_mean, axis=0)**2/np.sqrt(2* np.pi*np.prod(normal3_std)))
normal4 = np.exp(-np.linalg.norm(xy-normal4_mean, axis=0)**2/np.sqrt(2* np.pi*np.prod(normal4_std)))

t1 = normal1 + normal2
t2 = normal3 + normal4
t1 = t1/t1.sum()
t2 = t2/t2.sum()

t1_flat = t1.flatten()
t2_flat = t2.flatten()

xy_flat = xy.reshape(2,-1)
dist = np.linalg.norm(xy_flat[:,:,None]-xy_flat[:,None], axis=0)
# a = np.array([0.5, 0.5])  # 分布a
# b = np.array([0.5, 0.5])  # 分布b

# # コスト行列
# M = np.array([[0, 1],
#               [1, 0]])

# 最適輸送問題を解く
ot_plan = ot.emd(t1_flat, t2_flat, dist, numThreads="max")

print("Optimal transport plan:")
print(ot_plan.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# サーフェスプロット
ax.plot_surface(x, y, t1*1000, cmap='magma')
ax.plot_surface(x, y, t2*1000, cmap='viridis')

# ラベルの設定
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

match_idx = np.argmax(ot_plan, axis=-1)
match_max = np.max(ot_plan, axis=-1)

match_xy = xy_flat[:, match_idx]
z = np.zeros(match_xy.shape[1])
norm = (xy_flat - match_xy)**2
norm = np.sqrt(norm.sum(axis=0))
# flag = (norm < 2*2**0.5) & (norm > 2)
flag = (norm > 0.3) & (match_max != 0) & (match_idx % 16 == 0) & (t1_flat > 1e-6)
match_xy = match_xy - xy_flat

colors = plt.cm.jet(match_max / np.max(match_max))

dist = ax.quiver(xy_flat[0, flag], xy_flat[1, flag], z[flag]+2, match_xy[0, flag], match_xy[1, flag], z[flag], color=colors[flag])

# 表示
plt.savefig("pot_test.png")
# カメラの視点を更新する関数
def update(frame):
    ax.view_init(elev=30, azim=frame)
    return dist,

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False)

# 動画として保存
ani.save('quiver_rot.mp4', writer='ffmpeg', fps=30)