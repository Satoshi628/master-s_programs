import numpy as np
from itertools import product
from pulp import *

import os
from contextlib import redirect_stdout

def problem_solve():
    N = 10
    M = 10
    p = np.random.rand(N)
    C = np.random.randint(0,2,[M,N])

    problem = LpProblem(sense=LpMaximize)

    idx = [LpVariable(f"{i}", cat=LpBinary) for i in range(N)]

    problem += lpSum(p[i]*idx[i] for i in range(N))
    for i in range(M):
        problem += lpSum(C[i,j]*idx[j] for j in range(N)) == 1
    
    #解を求める
    problem.solve()

    onehot = np.array([item.value() for item in idx])
    print(p)
    print(C)
    print(onehot)
    print((p*onehot).sum())
    print(LpStatus[problem.status])




if __name__ == "__main__":
    problem_solve()

