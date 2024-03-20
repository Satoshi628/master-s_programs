import os
import math
import faiss
import umap

import glob
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation


k = 3

def use_image_hist():
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


cate=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
def feature_map():
    category = "capsule"
    few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Kshot_Approx/4-Shot_0"
    full_data_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Few/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0_46"

    few_shot_path = os.path.join(few_shot_path, "models", f"mvtec_{category}", "nnscorer_search_index.faiss")
    full_data_path = os.path.join(full_data_path, "models", f"mvtec_{category}", "nnscorer_search_index.faiss")
    
    index = faiss.read_index(few_shot_path)
    few_vector = index.reconstruct_n(0, index.ntotal)

    index = faiss.read_index(full_data_path)
    full_vector = index.reconstruct_n(0, index.ntotal)

    render = umap.UMAP()
    render.fit(full_vector)
    few_emb = render.transform(few_vector)
    full_emb = render.transform(full_vector)

    #最初にfullをやり、fullは背面fewは前面にやる。
    plt.scatter(full_emb[:,0], full_emb[:,1], s=3, c="b", label="full data")
    plt.scatter(few_emb[:,0], few_emb[:,1], s=3, c="r", label="few data")
    plt.savefig(f'images/_fewshot_umap_{category}.png')
    plt.close()


cate=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
def feature_map_video():
    category = "pill"
    few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data/multi_time_sample_0"
    data_splits = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
    data_splits = [data/ max(data_splits) for data in data_splits]
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_splits)))  # フレームごとに使用する色のリスト
    few_shot_path = os.path.join(few_shot_path, "models", f"mvtec_{category}", "nnscorer_search_index.faiss")
    
    index = faiss.read_index(few_shot_path)
    few_vector = index.reconstruct_n(0, index.ntotal)
    print(few_vector.shape)

    render = umap.UMAP()
    render.fit(few_vector)
    few_emb = render.transform(few_vector)

    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_splits)))  # フレームごとに使用する色のリスト

    # アニメーションの初期化関数
    def init():
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        return fig,

    # アニメーション更新関数
    def update(frame):
        start = int(data_splits[frame] * len(few_emb))
        end = int(data_splits[frame + 1] * len(few_emb))
        ax.scatter(few_emb[start:end, 0], few_emb[start:end, 1], color=colors[frame])
        return fig,
    # アニメーションを作成
    ani = FuncAnimation(fig, update, frames=range(len(data_splits)-1), init_func=init, blit=False)

    # アニメーションを表示（または保存）
    writergif = animation.PillowWriter(fps=1)
    ani.save(f'_feature_map_{category}.gif', writer=writergif)

def LOF_video():
    category = "bottle"
    few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data/multi_time_sample_1"
    data_splits = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
    data_splits = [data/ max(data_splits) for data in data_splits]

    few_shot_path = os.path.join(few_shot_path, "models", f"mvtec_{category}", "nnscorer_search_index.faiss")
    
    index = faiss.read_index(few_shot_path)
    few_vector = index.reconstruct_n(0, index.ntotal)
    print(few_vector.shape)
    
    
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_splits)))  # フレームごとに使用する色のリスト

    # アニメーションの初期化関数
    def init():
        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 1)
        return fig,

    # アニメーション更新関数
    def update(frame):
        start = int(data_splits[frame] * len(few_vector))
        end = int(data_splits[frame + 1] * len(few_vector))
        
        vector = few_vector[:end]
        
        #LOF
        clf = LocalOutlierFactor(n_neighbors=int(k), metric='l2')
        clf.fit(vector)
        scores = -clf.negative_outlier_factor_

        ax.hist(scores[start:], bins=20, alpha=0.2)
        return fig,
    # アニメーションを作成
    ani = FuncAnimation(fig, update, frames=range(len(data_splits)-1), init_func=init, blit=False)

    # アニメーションを表示（または保存）
    writergif = animation.PillowWriter(fps=1)
    ani.save(f'_LOF_video_{category}.gif', writer=writergif)


def LOF_graph():
    category = "bottle"
    few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data/multi_time_sample_1"
    # few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data/multi_time_sample"
    data_splits = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
    data_splits = [data/ max(data_splits) for data in data_splits]

    few_shot_path = os.path.join(few_shot_path, "models", f"mvtec_{category}", "nnscorer_search_index.faiss")
    
    index = faiss.read_index(few_shot_path)
    few_vector = index.reconstruct_n(0, index.ntotal)
    print(few_vector.shape)
    
    
    fig, ax = plt.subplots()
    scores = []
    anomaly_num = []
    anomaly_list = []
    for frame in range(len(data_splits)-1):
        start = int(data_splits[frame] * len(few_vector))
        end = int(data_splits[frame + 1] * len(few_vector))
        
        vector = few_vector[:end]
        
        #LOF
        clf = LocalOutlierFactor(n_neighbors=int(k), metric='l2')
        anomaly = clf.fit_predict(vector)[start:]
        score = -clf.negative_outlier_factor_[start:]
        scores.append(score)
        anomaly_list.append(anomaly)
        anomaly_num.append((anomaly == -1).sum()/ len(anomaly))

    data_splits = np.array(data_splits[1:])
    anomaly_num = np.array(anomaly_num)
    scores_max = np.array([s.max() for s in scores])
    scores_mean = np.array([np.mean(s) for s in scores])

    nonanomaly_score = np.array([np.sum(s[a!=-1])/((a!=-1).sum()+1e-7) for s, a in zip(scores, anomaly_list)])
    anomaly_scores = np.array([np.sum(s[a==-1])/((a==-1).sum()+1e-7) for s, a in zip(scores, anomaly_list)])
    ax.plot(data_splits, scores_max, label="max", color="red")
    ax.plot(data_splits, scores_mean, label="mean", color="blue")
    ax.plot(data_splits, nonanomaly_score, label="non anomaly", color="c")
    ax.plot(data_splits, anomaly_scores, label="anomaly", color="m")
    # ax.plot(data_splits, anomaly_num, label="number of anomaly", color="k")
    
    # x 軸のラベルを設定する。
    ax.set_xlabel("data rate")

    # y 軸のラベルを設定する。
    ax.set_ylabel("LOF")

    ax.legend()

    plt.savefig(f'_LOF_gprah_{category}.png')
    plt.close()


def LOF_feature_map():
    category = "pill"
    few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data/multi_time_sample_0"
    data_splits = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
    data_splits = [data/ max(data_splits) for data in data_splits]

    few_shot_path = os.path.join(few_shot_path, "models", f"mvtec_{category}", "nnscorer_search_index.faiss")
    
    index = faiss.read_index(few_shot_path)
    few_vector = index.reconstruct_n(0, index.ntotal)
    print(few_vector.shape)
    
    render = umap.UMAP()
    render.fit(few_vector)
    few_emb = render.transform(few_vector)
    
    scores = []
    for frame in range(len(data_splits)-1):
        start = int(data_splits[frame] * len(few_vector))
        end = int(data_splits[frame + 1] * len(few_vector))
        
        vector = few_vector[:end]
        
        #LOF
        clf = LocalOutlierFactor(n_neighbors=int(k), metric='l2')
        anomaly = clf.fit_predict(vector)
        score = -clf.negative_outlier_factor_[start:]
        scores.append(score)

    data_splits = np.array(data_splits[1:])
    scores = np.concatenate(scores)
    # scores = np.log(scores)
    scores = (scores-scores.min())/(scores.max()-scores.min())

    
    fig, ax = plt.subplots()
    # アニメーション更新関数
    def update(frame):
        start = int(data_splits[frame] * len(few_emb))
        end = int(data_splits[frame + 1] * len(few_emb))
        ax.scatter(few_emb[start:end,0], few_emb[start:end,1], c=scores[start:end], s=20, vmin=0., vmax=1., cmap="viridis")

        return fig,
    # アニメーションを作成
    ani = FuncAnimation(fig, update, frames=range(len(data_splits)-1), blit=False)
    
    # アニメーションを表示（または保存）
    writergif = animation.PillowWriter(fps=1)
    ani.save(f'_LOF_feature_{category}.gif', writer=writergif)
    plt.close()



def LOF_scatter():
    cate=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'transistor', 'wood', 'zipper']
    # cate=['cable', 'capsule', 'grid', 'metal_nut', 'pill', 'screw', 'transistor']
    # cate=['capsule', 'grid',]
    colors = ["b", "c", "g", "m", "r", "y", "cyan", "orange", "lime", "skyblue", "k", "purple", "violet", "pink"]
    few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data"
    result_paths = sorted(glob.glob(os.path.join(few_shot_path, "*", "results.csv")))
    print(result_paths)

    data_splits = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
    data_splits = [data/ max(data_splits) for data in data_splits]

    
    #結果収集
    results = []
    for path in result_paths:
        result = np.loadtxt(path, delimiter=",", dtype=str)

        #ヘッド削除
        result = result[1:]
        #Mean削除
        result = result[:-1]
        results.append(result)
    result = np.concatenate(results)

    cate_name = result[:,0]
    data_split = result[:,1].astype(np.float32)
    change_distance = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)

    all_Idiff = []
    all_score = []
    fig, ax = plt.subplots()
    for c, color in zip(cate, colors):
        print(c)
        path = glob.glob(os.path.join(few_shot_path, "*", "models", f"mvtec_{c}", "nnscorer_search_index.faiss"))[0]
        index = faiss.read_index(path)
        few_vector = index.reconstruct_n(0, index.ntotal)
        print(few_vector.shape)

        flag = np.array([c in name for name in cate_name])
        I = I_AUROC[flag]

        I_diff_list = []
        score_list = []
        min_auroc = 99999.
        for frame in range(len(data_splits)-1):
            start = int(data_splits[frame] * len(few_vector))
            end = int(data_splits[frame + 1] * len(few_vector))
            if min_auroc > I[frame]:
                min_auroc = I[frame]
            i_diff = I[frame] - min_auroc
            vector = few_vector[:end]
            
            #LOF
            clf = LocalOutlierFactor(n_neighbors=int(k), metric='l2')
            anomaly = clf.fit_predict(vector)
            score = -clf.negative_outlier_factor_
            I_diff_list.append(i_diff)
            # score_list.append(score.max())
            # score_list.append(np.mean(score))
            # score_list.append(score[anomaly!=-1].sum()/((anomaly!=-1).sum()+1e-7))
            score_list.append((anomaly==-1).sum())
        ax.scatter(I_diff_list, score_list, label=c, color=color)
        all_Idiff.append(np.array(I_diff_list))
        all_score.append(np.array(score_list))

    # x 軸のラベルを設定する。
    ax.set_xlabel("I-AUROC[%]")

    # y 軸のラベルを設定する。
    ax.set_ylabel("LOF")

    ax.legend()

    plt.savefig(f'_LOF_scatter_all_mean.png')
    plt.close()
    
    all_Idiff = np.concatenate(all_Idiff)
    all_score = np.concatenate(all_score)
    print("相関係数",np.corrcoef(all_Idiff, all_score))




def distance_scatter():
    cate=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'transistor', 'wood', 'zipper']
    colors = ["b", "c", "g", "m", "r", "y", "cyan", "orange", "lime", "skyblue", "k", "purple", "violet", "pink"]
    few_shot_path = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Data"
    result_paths = sorted(glob.glob(os.path.join(few_shot_path, "*", "results.csv")))
    print(result_paths)

    
    #結果収集
    results = []
    for path in result_paths:
        result = np.loadtxt(path, delimiter=",", dtype=str)

        #ヘッド削除
        result = result[1:]
        #Mean削除
        result = result[:-1]
        print(result.shape)
        results.append(result)
    result = np.concatenate(results)

    cate_name = result[:,0]
    data_split = result[:,1].astype(np.float32)
    change_distance = result[:,2].astype(np.float32)
    I_AUROC = result[:,3].astype(np.float32)


    fig, ax = plt.subplots()
    all_I_diff_list = []
    dist_list = []
    for c, color in zip(cate, colors):
        print(c)

        flag = np.array([c in name for name in cate_name])
        I = I_AUROC[flag]
        Dist = change_distance[flag]

        dist = Dist[1:]
        I_diff = np.diff(I)
        
        flag = dist == 0.
        dist = dist[~flag]
        I_diff = I_diff[~flag]

        all_I_diff_list.append(I_diff)
        dist_list.append(dist)

        ax.scatter(I_diff, dist, label=c, color=color)
    
    # x 軸のラベルを設定する。
    ax.set_xlabel("I-AUROC[%]")

    # y 軸のラベルを設定する。
    ax.set_ylabel("distance")

    ax.legend()

    plt.savefig(f'_distance_scatter_all.png')
    plt.close()

    all_I_diff_list = np.concatenate(all_I_diff_list)
    dist_list = np.concatenate(dist_list)
    print("相関係数",np.corrcoef(all_I_diff_list, dist_list)[0,1])

# feature_map_video()
# LOF_video()
# LOF_graph()
# LOF_feature_map()
LOF_scatter()
# distance_scatter()
# feature_map()
