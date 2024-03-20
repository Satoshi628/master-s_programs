import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def all_category_memory():
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Memory/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0_2"
    cate = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 'pill','screw','tile','transistor','wood','zipper']
    colors = ["b", "c", "g", "m", "r", "y", "cyan", "orange", "lime", "skyblue", "k", "purple", "violet", "pink"]
    result_path = os.path.join(save_dir, "results.csv")
    result = np.loadtxt(result_path, delimiter=",", dtype=str)

    #ヘッド削除
    result = result[1:]
    #Mean削除
    result = result[:-1]

    cate_name = result[:,0]
    parcents = result[:,1].astype(np.float32)
    run_time = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)

    fig, ax = plt.subplots()

    for c, color in zip(cate, colors):
        flag = cate_name == c
        p = parcents[flag]
        I = I_AUROC[flag]
        I = run_time[flag]

        ax.plot(p, I, color, label=c)

    # plt.ylim(0.925, 1.0)

    # x 軸のラベルを設定する。
    ax.set_xlabel("Memory usage")

    # y 軸のラベルを設定する。
    ax.set_ylabel("I-AUROC[%]")
    
    # 凡例を表示
    plt.legend()

    plt.savefig(f"_runtime_all.png")
    input("owa")

def all_category_distance1():
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Memory/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0"
    cate = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut','leather', 'metal_nut', 'pill','screw','tile','transistor','wood','zipper']
    colors = ["b", "c", "g", "m", "r", "y", "cyan", "orange", "lime", "skyblue", "k", "purple", "violet", "pink"]
    result_path = os.path.join(save_dir, "results.csv")
    result = np.loadtxt(result_path, delimiter=",", dtype=str)

    #ヘッド削除
    result = result[1:]
    #Mean削除
    result = result[:-1]

    cate_name = result[:,0]
    parcents = result[:,1].astype(np.float32)
    run_time = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)

    fig, ax = plt.subplots()

    for c, color in zip(cate, colors):
        dist_path = os.path.join(save_dir, f"models/mvtec_{c}", "distance.npz")
        #データロード
        distance = np.load(dist_path)["arr_0"]
        # distance = np.gradient(distance)
        # distance = np.gradient(distance)

        flag = cate_name == f"mvtec_{c}"
        p = parcents[flag]
        I = I_AUROC[flag]

        dist = np.array([distance[int(_p*len(distance)-1)] for _p in p])
        # dist = dist/dist.max()
        ax.plot(dist, I, color, label=c)

        _dist = dist/distance.max()
        x = np.arange(len(distance))/len(distance)
        x = x[[int(_p*len(distance)-1) for _p in p]]
        center_dist = _dist**2 + x**2
        idx = center_dist.argmin()

        plt.scatter(dist[idx], I[idx], color=color)  # 点を赤色で表示
        # 点の値を注釈で表示
        # plt.annotate(f'({dist[idx]:.2%}, {I[idx]:.2f})', # 表示するテキスト
        #             (dist[idx], I[idx]),         # テキストを表示する位置
        #             textcoords="offset points", # テキストの位置の基準
        #             xytext=(0,10),              # テキストの座標からのオフセット
        #             ha='center')                # 水平方向のテキスト位置（ここでは中央）


    plt.ylim(0.925, 1.0)

    # x 軸のラベルを設定する。
    ax.set_xlabel("distance")

    # y 軸のラベルを設定する。
    ax.set_ylabel("I-AUROC[%]")
    
    # 凡例を表示
    plt.legend()

    plt.savefig(f"_distance_all_max.png")



def category_distance():
    category = "grid"
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Memory2/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0"
    dist_path = os.path.join(save_dir, f"models/mvtec_{category}", "distance.npz")
    result_path = os.path.join(save_dir, "results.csv")
    result = np.loadtxt(result_path, delimiter=",", dtype=str)
    #データロード
    distance = np.load(dist_path)["arr_0"]

    #ヘッド削除
    result = result[1:]
    #Mean削除
    result = result[:-1]

    parcents = result[:,1].astype(np.float32)
    run_time = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)

    print(distance.shape)


    dist = np.array([distance[int(p*len(distance)-1)] for p in parcents])

    fig, ax = plt.subplots()

    # ヒストグラムを描画する。
    ax.plot(dist, I_AUROC)

    # x 軸のラベルを設定する。
    ax.set_xlabel("distance")

    # y 軸のラベルを設定する。
    ax.set_ylabel("I-AUROC[%]")


    plt.savefig(f"_distance_{category}.png")
    plt.close()


    fig, ax = plt.subplots()

    ax.plot(parcents, run_time)

    # x 軸のラベルを設定する。
    ax.set_xlabel("Memory usage")

    # y 軸のラベルを設定する。
    ax.set_ylabel("run time")

    plt.scatter(parcents[0], run_time[0], color='red')  # 点を赤色で表示
    # 点の値を注釈で表示
    plt.annotate(f'({parcents[0]:.2%}, {run_time[0]:.2f})', # 表示するテキスト
                (parcents[0], run_time[0]),         # テキストを表示する位置
                textcoords="offset points", # テキストの位置の基準
                xytext=(0,10),              # テキストの座標からのオフセット
                ha='center')                # 水平方向のテキスト位置（ここでは中央）

    plt.scatter(parcents[-1], run_time[-1], color='red')  # 点を赤色で表示
    # 点の値を注釈で表示
    plt.annotate(f'({parcents[-1]:.2%}, {run_time[-1]:.2f})', # 表示するテキスト
                (parcents[-1], run_time[-1]),         # テキストを表示する位置
                textcoords="offset points", # テキストの位置の基準
                xytext=(0,10),              # テキストの座標からのオフセット
                ha='center')                # 水平方向のテキスト位置（ここでは中央）



    plt.savefig(f"_runtime_{category}.png")
    plt.close()

def mean():
    parcent = 1.0

    result_path = os.path.join(save_dir, "results.csv")
    result = np.loadtxt(result_path, delimiter=",", dtype=str)

    #ヘッド削除
    result = result[1:]
    #Mean削除
    result = result[:-1]

    parcents = result[:,1].astype(np.float32)
    run_time = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)
    P_AUROC = result[:,4].astype(np.float32)
    PRO = result[:,5].astype(np.float32)

    flag = parcents == parcent
    run_time = run_time[flag].mean()
    I_AUROC = I_AUROC[flag].mean()
    P_AUROC = P_AUROC[flag].mean()
    PRO = PRO[flag].mean()
    print(run_time,
        I_AUROC,
        P_AUROC,
        PRO)

def dist_idx():
    category = "grid"
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Memory2/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0"
    # save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data/multi_time_sample"
    dist_path = os.path.join(save_dir, f"models/mvtec_{category}", "distance.npz")
    distance = np.load(dist_path)["arr_0"]

    idx = np.arange(len(distance))
    print(distance[:3])
    center_dist = (distance/max(distance))**2 + (idx/max(idx))**2
    point_idx = center_dist.argmin()
    dist = (distance-distance.min())/(distance.max()-distance.min())
    kotei_point_idx = np.abs(dist - 0.2).argmin()

    # center_dist = (distance)**2 + (idx)**2
    
    # 移動平均を計算
    # window_size = 21
    # distance = pd.Series(np.concatenate([*[distance[:1] for _ in range(window_size)],distance])).rolling(window=window_size).mean().values
    # distance = distance[window_size:]
    num = 500
    dist_idx_percentile = np.percentile(distance, np.arange(num+1)/num*100)
    dist_idx = np.abs(distance[None] - dist_idx_percentile[:,None]).argmin(axis=-1)[::-1]

    CS = CubicSpline(idx[dist_idx], distance[dist_idx], axis=0)
    grad1 = np.gradient(distance, idx)
    grad2 = np.gradient(grad1, idx)

    
    delta = 0.001
    times = np.arange(0, idx[-1], delta)
    line = CS(times)
    dxy_dt = np.gradient(line, times, axis=0)
    
    dxy_dt_2 = np.gradient(dxy_dt, times, axis=0)

    # R = ((dxy_dt ** 2).sum(axis=-1)) ** 1.5 / np.clip(np.abs(dxy_dt[:, 0] * dxy_dt_2[:, 1] - dxy_dt[:, 1] * dxy_dt_2[:, 0]), 1e-7, None)
    R = (1+dxy_dt**2)**1.5/(np.abs(dxy_dt_2) + 1e-6)



    fig, ax = plt.subplots()



    ax.plot(idx, distance, label="f(x)", color="blue")
    ax.plot(idx, center_dist, label="center distance", color="red")
    ax.plot([idx[kotei_point_idx],idx[kotei_point_idx]], [0.0,max(distance)], label="const0.2", color="k")
    ax.scatter(idx[point_idx], distance[point_idx], color="red")
    ax.scatter(idx[kotei_point_idx], distance[kotei_point_idx], color="blue")

    # ax.plot(times, dxy_dt, label="f'(x)", color="red")
    # ax.plot(times, dxy_dt_2, label="f''(x)", color="green")
    # ax.plot(times, line, label="f(x) spline", color="c")
    # ax.plot(times, R/10**6, label="R(x)", color="k")

    # x 軸のラベルを設定する。
    ax.set_xlabel("idx")

    # y 軸のラベルを設定する。
    ax.set_ylabel("distance")
    ax.legend()

    plt.savefig(f"_dist_idx_{category}.png")
    plt.close()


# all_category_memory()
# all_category_distance1()
# all_category_distance()
# mean()
# category_distance(category, )
dist_idx()

def all_category_memory_area():
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/Denso_Results_Memory/"
    cate=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    cate=['denso_dn_data3', 'denso_dn_data_new']
    colors = ["b", "c", "g", "m", "r", "y", "cyan", "orange", "lime", "skyblue", "k", "purple", "violet", "pink", "gold"]
    result_paths = glob.glob(os.path.join(save_dir, "*", "results.csv"))
    result = np.stack([np.loadtxt(path, delimiter=",", dtype=str) for path in result_paths], axis=-1)

    #ヘッド削除
    result = result[1:]
    #Mean削除
    result = result[:-1]

    cate_name = result[:,0]
    parcents = result[:,1].astype(np.float32)
    run_time = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)*100

    fig, ax = plt.subplots()

    pars = []
    stds = []

    for c, color in zip(cate, colors):
        flag = np.all(cate_name == ("mvtec_" + c), axis=-1)
        flag = np.all(cate_name ==  c, axis=-1)

        p = parcents[flag]
        I = I_AUROC[flag]
        # I = run_time[flag]
        p_mean = p.mean(axis=-1)
        
        I_mean = I.mean(axis=-1)
        I_std = np.std(I,axis=-1)

        ax.plot(p_mean, I_mean, color, label=c)
        ax.fill_between(p_mean, I_mean+I_std, I_mean-I_std, color=color, alpha=0.4)


    plt.ylim(92.5, 100)

    # x 軸のラベルを設定する。
    ax.set_xlabel("Memory usage")

    # y 軸のラベルを設定する。
    ax.set_ylabel("I-AUROC[%]")
    
    # 凡例を表示
    plt.legend()

    plt.savefig(f"images/_memory_all10_area.png")
    
    plt.close()


def all_category_memory_std():
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/Denso_Results_Memory/"
    save_dir2 = "/mnt/kamiya/code/patchcore-inspection/results/Denso_Results_Memory2/"
    cate=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    cate=['denso_dn_data3', 'denso_dn_data_new']
    colors = ["b", "c", "g", "m", "r", "y", "cyan", "orange", "lime", "skyblue", "k", "purple", "violet", "pink", "gold"]
    result_paths = glob.glob(os.path.join(save_dir, "*", "results.csv"))
    result = np.stack([np.loadtxt(path, delimiter=",", dtype=str) for path in result_paths], axis=-1)

    result_paths = glob.glob(os.path.join(save_dir2, "*", "results.csv"))
    result_ot = np.stack([np.loadtxt(path, delimiter=",", dtype=str) for path in result_paths], axis=-1)



    #ヘッド削除
    result = result[1:]
    #Mean削除
    result = result[:-1]

    cate_name = result[:,0]
    parcents = result[:,1].astype(np.float32)
    run_time = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)*100

    
    #ヘッド削除
    result_ot = result_ot[1:]
    #Mean削除
    result_ot = result_ot[:-1]

    cate_name_ot = result_ot[:,0]
    parcents_ot = result_ot[:,1].astype(np.float32)
    run_time_ot = result_ot[:,2].astype(np.float32)
    I_AUROC_ot = result_ot[:,3].astype(np.float32)*100

    fig, ax = plt.subplots()

    for c, color in zip(cate, colors):
        flag = np.all(cate_name == ("mvtec_" + c), axis=-1)
        flag = np.all(cate_name == c, axis=-1)

        p = parcents[flag]
        I = I_AUROC[flag]
        
        flag = np.all(cate_name_ot == ("mvtec_" + c), axis=-1)
        flag = np.all(cate_name_ot == c, axis=-1)
        p_ot = parcents_ot[flag].mean(axis=-1)
        I_ot = np.std(I_AUROC_ot[flag],axis=-1)
        # I = run_time[flag]
        p_mean = p.mean(axis=-1)
        
        I_mean = I.mean(axis=-1)
        I_std = np.std(I,axis=-1)

        p_mean = np.concatenate([p_mean, p_ot])
        I_std = np.concatenate([I_std, I_ot])
        sort_idx = np.argsort(p_mean)
        p_mean = p_mean[sort_idx]
        I_std = I_std[sort_idx]

        ax.plot(p_mean, I_std, color, label=c)
        
        plt.scatter(p_ot[0], I_ot[0], color=color)  # 点を赤色で表示
        # ax.fill_between(p_mean, I_mean+I_std, I_mean-I_std, alpha=0.4)

    # plt.ylim(0.925, 1.0)

    # x 軸のラベルを設定する。
    ax.set_xlabel("Memory usage")

    # y 軸のラベルを設定する。
    ax.set_ylabel("I-AUROC std")
    
    # 凡例を表示
    plt.legend()

    plt.savefig(f"images/_memory_all10_std.png")
    
    plt.close()

# all_category_memory_std()
# all_category_memory_area()