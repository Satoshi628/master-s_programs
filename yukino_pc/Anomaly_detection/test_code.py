import numpy as np
import torch
import lap
import time

n = 100
cost = np.random.rand(int(n*0.1),n)
start = time.time()
cost, x, y = lap.lapjv(cost, extend_cost=True)
end = time.time()
print(f"run time:{end-start} sec")

torch.rand([10,10])/np.array(2.5)