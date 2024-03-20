
import os
import glob
import numpy as np
result_path = "/mnt/kamiya/code/FastFlow/_VisA"

paths = sorted(glob.glob(os.path.join(result_path, "*", "_result.txt")))

category = [p.split("/")[-2].replace("exp_", "").replace("_K512", "") for p in paths]
result = np.stack([np.loadtxt(p, dtype=str, comments="%") for p in paths])
print(result.shape)
acc_name = result[0,:,0]
nums = result[:,:,1].astype(np.float32) / 100

with open(os.path.join(result_path, "_result.txt"), mode="w") as f:
    f.write("\t" + "\t".join(acc_name.tolist())+"\n")

    for cate, n in zip(category, nums):
        print(cate)
        f.write(cate + "\t" + "\t".join(n.astype(str).tolist()) + "\n")


    f.write("Mean\t" + "\t".join(nums.mean(0).astype(str).tolist()) + "\n")
    
