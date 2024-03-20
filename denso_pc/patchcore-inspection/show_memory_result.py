import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def save_std_result():
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Faster/"
    result_paths = glob.glob(os.path.join(save_dir, "*", "results.csv"))
    result = np.stack([np.loadtxt(path, delimiter=",", dtype=str) for path in result_paths], axis=-1)

    #ヘッド削除
    heads = result[0, :, 0]
    result = result[1:]
    #Mean削除
    result = result[:-1]

    cate_name = result[:,0]
    nums = result[:,1:].astype(np.float32)
    mean = nums.mean(axis=-1)
    std = np.std(nums, axis=-1)
    with open("result.txt", mode="w") as f:
        heads_text = "\t".join(heads.tolist()) + "\n"
        f.write(heads_text)
        for cate, l_mean, l_std in zip(cate_name, mean, std):
            text = f"{cate[0]}\t"
            for m, s in zip(l_mean, l_std):
                text += f"{m:.2%}±{s:.2%}\t"
            text += "\n"
            f.write(text)

def save_fewshot_std_result():
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Few"
    result_paths = glob.glob(os.path.join(save_dir, "*", "results.csv"))
    result = np.stack([np.loadtxt(path, delimiter=",", dtype=str) for path in result_paths], axis=-1)

    #ヘッド削除
    heads = result[0, :, 0]
    result = result[1:]
    #Mean削除
    result = result[:-1]

    cate_name = result[:,0, 0]
    data_rate = result[1,1].astype(np.float32)
    
    uni_data = np.unique(data_rate)

    nums = result[:,1:].astype(np.float32)

    mean_list = []
    std_list = []
    for data in uni_data:
        flag = data_rate == data

        rate_data = nums[:,:,flag]

        mean = rate_data.mean(axis=-1)
        std = np.std(rate_data, axis=-1)
        mean_list.append(mean)
        std_list.append(std)
    

    with open("result.txt", mode="w") as f:
        heads_text = "\t" + "\t".join(heads.tolist()) + "\n"
        f.write(heads_text)
        for _mean, std in zip(mean_list, std_list):
            for cate, l_mean, l_std in zip(cate_name, _mean, std):
                text = f"{cate}\t"
                for m, s in zip(l_mean, l_std):
                    text += f"{m:.2%}±{s:.2%}\t"
                text += "\n"
                f.write(text)

            text = "Mean\t"
            for m, s in zip(_mean.mean(axis=0), std.mean(axis=0)):
                text += f"{m:.2%}±{s:.2%}\t"
            text += "\n"
            f.write(text)


def save_kshot_std_result():
    cate = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Kshot_Approx"
    result_paths = sorted(glob.glob(os.path.join(save_dir, "8-Shot-*", "results.csv")))
    print(result_paths)

    k_shot_mean = []
    k_shot_std = []
    for path in result_paths:
        result = np.loadtxt(path, delimiter=",", dtype=str)

        #ヘッド削除
        heads = result[0, :]
        result = result[1:]
        #Mean削除
        result = result[:-1]

        cate_name = result[:,0]

        nums = result[:,1:].astype(np.float32)

        mean_list = []
        std_list = []
        for c in cate:
            flag = np.array([c in name for name in cate_name])

            cate_data = nums[flag,:]

            mean = cate_data.mean(axis=0)
            std = np.std(cate_data, axis=0)
            mean_list.append(mean)
            std_list.append(std)
        
        k_shot_mean.append(np.array(mean_list))
        k_shot_std.append(np.array(std_list))

    with open("result.txt", mode="w") as f:
        heads_text = "\t" + "\t".join(heads.tolist()) + "\n"
        f.write(heads_text)
        for _mean, std in zip(k_shot_mean, k_shot_std):
            for c, l_mean, l_std in zip(cate, _mean, std):
                text = f"{c}\t"
                for m, s in zip(l_mean, l_std):
                    text += f"{m:.2%}±{s:.2%}\t"
                text += "\n"
                f.write(text)

            text = "Mean\t"
            for m, s in zip(_mean.mean(axis=0), std.mean(axis=0)):
                text += f"{m:.2%}±{s:.2%}\t"
            text += "\n"
            f.write(text)


def save_kshot_std_result_wo_std():
    cate = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    save_dir = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Kshot_Approx"
    result_paths = sorted(glob.glob(os.path.join(save_dir, "8-Shot-*", "results.csv")))
    result_paths = sorted(glob.glob(os.path.join(save_dir, "4-Shot", "results.csv")))
    print(result_paths)

    k_shot_mean = []
    k_shot_std = []
    for path in result_paths:
        result = np.loadtxt(path, delimiter=",", dtype=str)

        #ヘッド削除
        heads = result[0, :]
        result = result[1:]
        #Mean削除
        result = result[:-1]

        cate_name = result[:,0]

        nums = result[:,1:].astype(np.float32)

        mean_list = []
        std_list = []
        for c in cate:
            flag = np.array([c in name for name in cate_name])

            cate_data = nums[flag,:]

            mean = cate_data.mean(axis=0)
            std = np.std(cate_data, axis=0)
            mean_list.append(mean)
            std_list.append(std)
        
        k_shot_mean.append(np.array(mean_list))
        k_shot_std.append(np.array(std_list))

    with open("result.txt", mode="w") as f:
        heads_text = "\t" + "\t".join(heads.tolist()) + "\n"
        f.write(heads_text)
        for _mean, std in zip(k_shot_mean, k_shot_std):
            for c, l_mean, l_std in zip(cate, _mean, std):
                text = f"{c}\t"
                for m, s in zip(l_mean, l_std):
                    text += f"{m:.2%}\t"
                text += "\n"
                f.write(text)

            text = "Mean\t"
            for m, s in zip(_mean.mean(axis=0), std.mean(axis=0)):
                text += f"{m:.2%}\t"
            text += "\n"
            f.write(text)


# save_fewshot_std_result()
save_kshot_std_result()
# save_std_result()
# save_kshot_std_result_wo_std()