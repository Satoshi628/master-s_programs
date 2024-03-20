import numpy as np

# カルマンフィルタの初期化
def initialize_filter(initial_state, initial_estimate_error):
    state = initial_state
    estimate_error = initial_estimate_error
    return state, estimate_error

# 予測ステップ
def predict(state, estimate_error, A, B, u, Q):
    # 予測
    predicted_state = np.dot(A, state) + np.dot(B, u)
    predicted_estimate_error = np.dot(np.dot(A, estimate_error), A.T) + Q
    return predicted_state, predicted_estimate_error

# 更新ステップ
def update(predicted_state, predicted_estimate_error, measurement, H, R):
    # カルマンゲインの計算
    kalman_gain = np.dot(np.dot(predicted_estimate_error, H.T), np.linalg.inv(np.dot(np.dot(H, predicted_estimate_error), H.T) + R))
    
    # 更新
    updated_state = predicted_state + np.dot(kalman_gain, (measurement - np.dot(H, predicted_state)))
    updated_estimate_error = np.dot((np.eye(len(state)) - np.dot(kalman_gain, H)), predicted_estimate_error)
    
    return updated_state, updated_estimate_error

# サンプルデータ
measurements = np.array([[1.2, 0.8],
                         [1.4, 0.7],
                         [1.7, 0.9],
                         [1.3, 0.6],
                         [1.9, 1.0]])

# カルマンフィルタのパラメータ
initial_state = np.array([1.0, 0.0])  # 初期状態 (位置と速度)
initial_estimate_error = np.eye(2)  # 初期推定誤差共分散行列
A = np.array([[1.0, 1.0], [0.0, 1.0]])  # 状態遷移行列
B = np.array([[0.5], [1.0]])  # 制御入力行列
u = np.array([0.1])  # 制御入力
Q = np.eye(2) * 0.01  # プロセスノイズの共分散行列
H = np.eye(2)  # 観測行列
R = np.eye(2) * 0.1  # 観測ノイズの共分散行列

# カルマンフィルタの初期化
state, estimate_error = initialize_filter(initial_state, initial_estimate_error)

# フィルタリング
filtered_states = []
for measurement in measurements:
    # 予測ステップ
    predicted_state, predicted_estimate_error = predict(state, estimate_error, A, B, u, Q)
    
    # 更新ステップ
    state, estimate_error = update(predicted_state, predicted_estimate_error, measurement, H, R)
    
    filtered_states.append(state)

print("フィルタリング結果:", filtered_states)
