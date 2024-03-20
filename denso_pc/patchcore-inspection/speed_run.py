import time
import faiss
import numpy as np
import torch
import matplotlib.pyplot as plt

dimension = 1024
H = 224 // 8
W = 224 // 8
N = 2
Run_Num = 100
device = "cuda:0"

def dist_map(feature, memory):
    f_times_f = (feature**2).sum(dim=-1)
    m_times_m = (memory**2).sum(dim=-1)
    f_times_m = feature.mm(memory.T)
    score = (f_times_f[:,None] + m_times_m[None] - 2*f_times_m).min(dim=-1)[0]
    return score.sqrt()

def faiss_run(index, feature):
    score, _ = index.search(feature.cpu().numpy(), 1)
    return score[:,0]

def reset_faiss(memory):
    #CPU
    # index = faiss.IndexFlatL2(dimension)
    #GPU
    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig())
    index.add(memory.cpu().numpy())
    return index


#CPU
# index = faiss.IndexFlatL2(dimension)
#GPU
index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig())

feature = torch.rand([N*H*W, dimension],device=device)
memory = torch.rand([100, dimension],device=device)
#最初の実行は遅くなる
faiss_run(index, feature)
dist_map(feature, memory)

memory_num = list(range(100,1000,100)) + list(range(1000,10000,1000)) #+ list(range(10000, 100001,10000))

dist_map_speed = []
faiss_speed = []
for num in memory_num:
    faiss_fps = 0.
    dist_fps = 0.
    for _ in range(Run_Num):
        feature = torch.rand([N*H*W, dimension],device=device)
        memory = torch.rand([num, dimension],device=device)
        index = reset_faiss(memory)

        #faiss run
        start_time = time.time()
        faiss_run(index, feature)
        end_time = time.time()
        faiss_fps += end_time - start_time

        
        #dist map run
        start_time = time.time()
        dist_map(feature, memory)
        end_time = time.time()
        dist_fps += end_time - start_time
    
    print(f"Memory {num} Faiss speed:{faiss_fps/Run_Num*1000:.2f}ms")
    faiss_speed.append(faiss_fps/Run_Num)
    print(f"Memory {num} dist map speed:{dist_fps/Run_Num*1000:.2f}ms")
    dist_map_speed.append(dist_fps/Run_Num)

memory_num = np.array(memory_num)
faiss_speed = np.array(faiss_speed)
dist_map_speed = np.array(dist_map_speed)

fig, ax = plt.subplots()

ax.plot(memory_num, faiss_speed*1000, label="faiss spped", color="red")
ax.plot(memory_num, dist_map_speed*1000, label="dist map spped", color="blue")

ax.legend()
# x 軸のラベルを設定する。
ax.set_xlabel("memory")

# y 軸のラベルを設定する。
ax.set_ylabel("run time[ms]")

plt.savefig(f"images/run_time.png")
plt.close()

