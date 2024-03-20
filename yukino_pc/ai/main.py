#---------------------------------#
#       ライブラリインポート      #
#---------------------------------#
# 仮想ディスプレイ
from pyvirtualdisplay import Display
# 仮想ディスプレイ起動後でないとインポートできないライブラリがあるため
# 最初に起動しておく
d = Display()
d.start()

# OpenAI Gym関連
import gym

# DeepLearning関連
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, concatenate
import tensorflow as tf

# DQN関連
import rl
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam
from rl.processors import MultiInputProcessor

# その他
import numpy as np
import matplotlib.pyplot as plt
# OpenAI Gymアップデートによる警告関連
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)

#---------------------------------#
#     シミュレーション環境設定    #
#---------------------------------#
## 作成したシミュレーション環境設定(同一ランタイム中のファイル変更可能性のためリロード)
import myenv
from myenv import MyEnv

ENV_NAME = "pysim2d-v0"
env = gym.make(ENV_NAME)
o = env.reset()

#---------------------------------#
#          DQNモデル設定          #
#---------------------------------#
# 入出力層のパラメータ数設定
window_length = 1
input_shape = (window_length,) + (5,)
nb_actions = 9

## DNN設計
# 入力層
c = input_ = Input(input_shape)
c = Flatten()(c)
# 中間層
# ToDo：層の深さ、ノードの数の決定
c = Dense(256, activation='relu')(c)
c = Dense(256, activation='relu')(c)
c = Dense(256, activation='sigmoid')(c)
# 出力層
c = Dense(nb_actions, activation='linear')(c)
# モデル化
model = Model(input_, c)

# DNNサマリ出力
print(model.summary())

#---------------------------------#
#          DQNagent設定           #
#---------------------------------#
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = BoltzmannQPolicy()
test_policy = GreedyQPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=512,\
                 target_model_update=1e-2, policy=policy, test_policy=test_policy, delta_clip=1, batch_size=64)
agent.compile(Adam(lr=1e-3))

## 学習回数の設定
# 総学習ステップは nb_train_steps × nb_loops
nb_train_steps = 10000  # 1ループ当たりのステップ数
nb_loops = 15  # ループ回数

## 転移学習用（DNN重みの初期値読み込み）
# 重みランダム初期値から学習する際はコメントアウトしておく
#preweight_path = "./results/try_001/weights_last.h5f".format()
#weights = agent.load_weights(preweight_path)

#---------------------------------#
#             DQN学習             #
#---------------------------------#
nb_valid_episodes = 7  # 評価パターン数

# GPUをアサインできているか確認
print(tf.test.gpu_device_name())

import os
# resultsフォルダをつくる（なければ）
if not os.path.isdir("./results"):
    os.makedirs("./results")

# weightsフォルダ内にユニーク名を作る
dir_list = os.listdir("./results")
dir_name = "try_"+str(len(dir_list)).zfill(3)
os.makedirs("./results/"+dir_name)

# 開始時刻記録ファイル
import pathlib
import datetime
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
pathlib.Path("./results/"+dir_name+'/'+now.strftime('%Y%m%d_%H%M%S')+"_begin").touch()

# myenvをコピー
import shutil
shutil.copy("./myenv.py", "./results/"+dir_name+"/myenv.py")

## コールバック関数設定(同一ランタイム中のファイル変更可能性のためリロード)
import envs.util
from envs.util import TrainLogger, TestLogger, plot_train_logger, plot_test_logger
cb_train_logger = TrainLogger()  # 学習ログ
cb_test_logger = TestLogger()    # 評価ログ

train_callbacks = [cb_train_logger]
test_callbacks = [cb_test_logger]


# DQN学習と検証
for i in range(nb_loops):
    # 学習
    print('loop start ' + str(i) + ' / ' + str(nb_loops-1))
    agent.training = True
    env.set_training()
    agent.fit(env, nb_steps=nb_train_steps, visualize=False, verbose=1, callbacks=train_callbacks, log_interval=nb_train_steps)

    # 評価パターン実行
    agent.training = False
    env.set_validation()
    env.reset_valid_ptn()
    cb_test_logger.reset()
    agent.test(env, nb_episodes=nb_valid_episodes, visualize=False, callbacks=test_callbacks)
    cb_test_logger.calc_test_result(nb_valid_episodes)

    # プロット
    figure1 = plot_train_logger(cb_train_logger, nb_loops)
    figure2 = plot_test_logger(cb_test_logger, nb_loops)

    # 学習結果の指標
    print('goaled      : ' + str(cb_train_logger.sum_list[0][-1]))
    print('collision   : ' + str(cb_train_logger.sum_list[1][-1]))
    print('step_over   : ' + str(cb_train_logger.sum_list[2][-1]))
    print('minus_reward: ' + str(cb_train_logger.sum_list[3][-1]))
    
    print("Goal rate: " + '{:.1f}'.format(cb_test_logger.goal_rate) + "[%]")
    print('total takt time: ' + '{:.1f}'.format(cb_test_logger.total_takt_time_list[i]) + '[s]')

    # 重み保存
    weights = agent.save_weights("./results/{}/weights_{}.h5f".format(dir_name, str(i).zfill(2)), overwrite=True)

    print('---- loop end ----')
    print('')

weights = agent.save_weights("./results/{}/weights_last.h5f".format(dir_name), overwrite=True)
figure1.savefig("./results/{}/training_result.jpg".format(dir_name), overwrite=True)
figure2.savefig("./results/{}/test_result.jpg".format(dir_name), overwrite=True)

env.close()

# 終了時刻記録ファイル
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
pathlib.Path("./results/"+dir_name+'/'+now.strftime('%Y%m%d_%H%M%S')+"_end").touch()
