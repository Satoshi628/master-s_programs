import numpy as np
from itertools import product
from pulp import *

import os
from contextlib import redirect_stdout

def problem_solve(Image_class_matrix, T, error_rate=0.05):
    """スプリット作成関数。error rateによってすこし分け方を変化させることができるかも

    Args:
        Image_class_matrix (numpy.ndarryay): 一つの施設の大きな画像からcropされるclass0,class1の画像数の行列
        T (numpy.ndarray): 1つの施設の画像をどの数で分けるかを示す行列
        error_rate (float, optional): 許容するエラー率.(M-T)/T.低いすぎると解がない可能性がある. Defaults to 0.05.

    Raises:
        TypeError: Image_class_matrixはnumpy.ndarrayでなければなりません
        TypeError: Tはnumpy.ndarrayでなければなりません
        ValueError: Image_class_matrixは2次元配列でなければなりません
        ValueError: Image_class_matrixの二次元目の大きさは2でなければなりません
        ValueError: Tは二次元配列でなければなりません
        ValueError: Tの大きさは[3,2]でなければなりません

    Returns:
        numpy.ndarray: クラスわけのone hot vector.大きさは[num,3]
    """    
    if not isinstance(Image_class_matrix, np.ndarray):
        raise TypeError("Image_class_matrixはnumpy.ndarrayでなければなりません。")
    if not isinstance(T, np.ndarray):
        raise TypeError("Tはnumpy.ndarrayでなければなりません。")

    class_number_size = Image_class_matrix.shape
    target_number_size = T.shape

    if len(class_number_size) != 2:
        raise ValueError("Image_class_matrixの次元数は2次元でなければなりません。")
    if class_number_size[1] != 2:
        raise ValueError("Image_class_matrixの大きさは[画像数,2]でなければなりません。")
    if len(target_number_size) != 2:
        raise ValueError("Tの次元数は2次元でなければなりません。")
    if target_number_size[0] != 3 or target_number_size[1] != 2:
        raise ValueError("Tの大きさは[3,2]でなければなりません。")

    problem = LpProblem()
    #idxは[100,3]の大きさを持ち、3次元のone hot vectorである。
    idx = [[LpVariable("{},{}".format(i, j), cat=LpBinary) for j in range(3)] for i in range(class_number_size[0])]
    #train,val,testの誤差を定義
    train_error_0 = LpVariable("train error 0", lowBound=0)
    val_error_0 = LpVariable("val error 0", lowBound=0)
    test_error_0 = LpVariable("test error 0", lowBound=0)
    train_error_1 = LpVariable("train error 1", lowBound=0)
    val_error_1 = LpVariable("val error 1", lowBound=0)
    test_error_1 = LpVariable("test error 1", lowBound=0)

    #目的関数:エラーが少なくなるように学習させる。
    #本来の目的関数はL1正則化
    problem += train_error_0 + val_error_0 + test_error_0 + train_error_1 + val_error_1 + test_error_1

    #制約条件
    for i in range(class_number_size[0]):
        #one hot vectorになるように制約を加える
        problem += lpSum(idx[i]) == 1

    #本来の目的関数が絶対値になるようにする。
    #train
    problem += lpSum(Image_class_matrix[i, 0] * idx[i][0] for i in range(class_number_size[0])) - T[0, 0] <= train_error_0
    problem += lpSum(Image_class_matrix[i, 1] * idx[i][0] for i in range(class_number_size[0])) - T[0, 1] <= train_error_1
    problem += lpSum(Image_class_matrix[i, 0] * idx[i][0] for i in range(class_number_size[0])) - T[0, 0] >= -train_error_0
    problem += lpSum(Image_class_matrix[i, 1] * idx[i][0] for i in range(class_number_size[0])) - T[0, 1] >= -train_error_1
    #val
    problem += lpSum(Image_class_matrix[i, 0] * idx[i][1] for i in range(class_number_size[0])) - T[1, 0] <= val_error_0
    problem += lpSum(Image_class_matrix[i, 1] * idx[i][1] for i in range(class_number_size[0])) - T[1, 1] <= val_error_1
    problem += lpSum(Image_class_matrix[i, 0] * idx[i][1] for i in range(class_number_size[0])) - T[1, 0] >= -val_error_0
    problem += lpSum(Image_class_matrix[i, 1] * idx[i][1] for i in range(class_number_size[0])) - T[1, 1] >= -val_error_1
    #test
    problem += lpSum(Image_class_matrix[i, 0] * idx[i][2] for i in range(class_number_size[0])) - T[2, 0] <= test_error_0
    problem += lpSum(Image_class_matrix[i, 1] * idx[i][2] for i in range(class_number_size[0])) - T[2, 1] <= test_error_1
    problem += lpSum(Image_class_matrix[i, 0] * idx[i][2] for i in range(class_number_size[0])) - T[2, 0] >= -test_error_0
    problem += lpSum(Image_class_matrix[i, 1] * idx[i][2] for i in range(class_number_size[0])) - T[2, 1] >= -test_error_1

    #errorの閾値を決める。目標値の±error rate%までは許容する
    #train class0
    problem += train_error_0 >= T[0, 0] * error_rate
    problem += train_error_0 <= T[0, 0] * (1 + error_rate)
    #train class1
    problem += train_error_1 >= T[0, 1] * error_rate
    problem += train_error_1 <= T[0, 1] * (1 + error_rate)
    #val class0
    problem += val_error_0 >= T[1, 0] * error_rate
    problem += val_error_0 <= T[1, 0] * (1 + error_rate)
    #val class1
    problem += val_error_1 >= T[1, 1] * error_rate
    problem += val_error_1 <= T[1, 1] * (1 + error_rate)
    #test class0
    problem += test_error_0 >= T[2, 0] * error_rate
    problem += test_error_0 <= T[2, 0] * (1 + error_rate)
    #test class1
    problem += test_error_1 >= T[2, 1] * error_rate
    problem += test_error_1 <= T[2, 1] * (1 + error_rate)

    #解を求める
    problem.solve()

    one_hot_vector = np.array([[item2.value() for item2 in item1] for item1 in idx])
    class0 = one_hot_vector * Image_class_matrix[:, 0, None]
    class0 = np.sum(class0, axis=0)
    class1 = one_hot_vector * Image_class_matrix[:, 1, None]
    class1 = np.sum(class1, axis=0)
    
    print(LpStatus[problem.status])
    print("目的関数", value(problem.objective))
    print("目標class0", T_0)
    print("目標class1",T_1)
    print("解class0",class0)
    print("解class1", class1)

    return one_hot_vector

#大きい画像の数
N = 100
n_max = 100
n_min = 0
#許容がないと計算に時間がかかる ±5%まで許容
error_rate = 0.05

#class_num作成、大きさは[大きい画像の数,2]
#class_num[:,0]はcropされたclass0の画像の枚数
#class_num[:,1]はcropされたclass0の画像の枚数
class_num = np.random.randint(n_min, n_max, (N, 2))

#Train,Val,Testの目標値
T_0 = np.array([np.sum(class_num[:, 0]) * 0.7, np.sum(class_num[:, 0]) * 0.1, np.sum(class_num[:, 0]) * 0.2])
T_1 = np.array([np.sum(class_num[:, 1]) * 0.7, np.sum(class_num[:, 1]) * 0.1, np.sum(class_num[:, 1]) * 0.2])

#[3,2]の大きさにする
#3はtrain,val,testを示している
#2はclass0,class1を示している
#[train class0, val class0, test class0]
#[train class1, val class1, test class1]
#という風に定義
T = np.stack([T_0, T_1], axis=-1)

with redirect_stdout(open(os.devnull, 'w')):
    vector = problem_solve(class_num, T, error_rate=0.01)

