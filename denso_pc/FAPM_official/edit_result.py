import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def save_std_result():
    save_dir = "/mnt/kamiya/code/FAPM_official/memory"
    result_paths = sorted(glob.glob(os.path.join(save_dir, "*.txt")))
    result = np.stack([np.loadtxt(path, delimiter="\t", dtype=str) for path in result_paths], axis=-1)

    #ヘッド削除
    heads = result[0, :, 0]
    result = result[1]

    cate_name = result[0]
    nums = result[1:].astype(np.float32)

    with open("result.txt", mode="w") as f:
        heads_text = "\t".join(heads.tolist()) + "\n"
        f.write(heads_text)
        for cate, num in zip(cate_name.T, nums.T):
            text = f"{cate}\t"
            for n in num:
                text += f"{n:.2%}\t"
            text += "\n"
            f.write(text)

save_std_result()