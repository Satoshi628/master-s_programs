from tifffile import TiffFile
import numpy as np
from pulp import *

path = "/mnt/kamiya/dataset/SCLIM2M/02G-5000.tif"

def load_3DSCLIM2M(path):
    # TIFFファイルを読み込む
    with TiffFile(path) as tif:
        #なぜかこれを実行しないとすべてを取得できない
        print(tif.flags)

        images = tif.asarray()

    # 9894 = (51 * 194, 384, 384)
    images = images.reshape(51, -1, 384, 384).transpose(0, 2, 3, 1)
    return images


N = 5
M = 6
cost_matrix = np.random.rand(N, M)
A = np.random.rand(N)
B = np.random.rand(M)
# AとBの総量を同じにする
A = A/A.sum()
B = B/B.sum()

problem = LpProblem(sense=LpMinimize)

P = [[LpVariable(f"[{col},{row}]", lowBound=0) for row in range(B.shape[0])] for col in range(A.shape[0])]

# minimum
problem += lpSum([lpDot(p, c) for p, c in zip(P, cost_matrix)])

for idx, P_i in enumerate(P):
    problem += lpSum(P_i) == A[idx]

for idx, (P_j) in enumerate(zip(*P)):
    problem += lpSum(P_j) == B[idx]


print("start solve")
problem.solve()
print(LpStatus[problem.status])

print(P)
result = []
for P_i in P:
    vec = []
    for P_ij in P_i:
        vec.append(P_ij.value())
    result.append(vec)

print(np.array(result))