import time
import faiss
import numpy as np
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap

# dimension = 256

# index = faiss.IndexFlatL2(dimension)

# vector = np.random.rand(100, dimension).astype(np.float32)
# idx = np.arange(100)

# index.add(vector)
# print(index.ntotal)
# print(index.code_size)
# vector_xb = index.reconstruct_n(0, 100)
# print(vector_xb.shape)

# search_vec = np.random.rand(10, dimension).astype(np.float32)
# distance, search_idx = index.search(serch_vec, 1)

# def self_distance_matrix(vector):
#     V_2 = np.matmul(vector[:, None], vector[:,:,None]).ravel()
#     vv = 2. * np.matmul(vector, vector.T)
#     return np.sqrt(np.clip(V_2[:,None] + V_2[None] - vv, 0., None))

# search_vec = np.random.rand(10, 16).astype(np.float32)



filename = "/mnt/kamiya/code/patchcore-inspection/results/MVTecAD_Results_Memory2/IM224_WR101_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/mvtec_bottle/nnscorer_search_index.faiss"
index = faiss.read_index(filename)
vector = index.reconstruct_n(0, index.ntotal)
print(vector.shape)
vector_idx = np.arange(vector.shape[0])

# distance_matrix = self_distance_matrix(vector)
# distance_matrix = np.tril(distance_matrix, k=-1)
# distance_matrix[distance_matrix==0] = 9999.
# distance_matrix = distance_matrix.min(axis=1)

# #最初の値は9999.になるため最大値に置き換え
# distance_matrix[0] = 0.
# distance_matrix[0] = distance_matrix.max()

# plt.plot(vector_idx, distance_matrix)
# plt.savefig('_distance.png')
# plt.close()

# embedding = umap.UMAP().fit_transform(vector)

# plt.scatter(embedding[:,0], embedding[:,1], s=3, c=vector_idx, cmap=cm.jet)
# plt.colorbar()
# plt.savefig('_umap.png')
# plt.close()
