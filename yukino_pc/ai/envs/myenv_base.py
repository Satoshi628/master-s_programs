import gym
import numpy as np
from gym import spaces
from .Viewer  import Viewer
from .PySim2D import PySim2D

class MyEnv_base(gym.Env):
    def __init__(self):
        # シミュレーション情報の初期化
        self.step_cnt = 0  # ステップ数カウント
        self.reward = 0    # 今回ステップの獲得報酬
        self.total_reward = 0  # 獲得総報酬（resetでクリア）
        self.done_index = 0   # どの理由でDoneになったかの情報（DQNコールバック用に定義）
        self.training = True  # 学習と検証の処理分け
        self.valid_ptn = 0    # 検証番号
        
        ## simulator
        self.simulator = PySim2D("envs/p_area.csv")
        
        ## gym space
        # observation space: [rel_pos_x, rel_pos_y, rel_pos_theta, speed, curvature]
        self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(5, ))
        # action space:
        self.action_space = spaces.Discrete(9)  # 0:そのまま 1:加速 2:減速
        
        # render
        self.viewer = None
        
    def reset(self, radius=5.5, phai=np.random.randint(low=-60, high=60)):
        self.step_cnt = 0  # ステップ数カウント
        self.reward = 0    # 今回ステップの獲得報酬
        self.total_reward = 0  # 獲得総報酬（resetでクリア）
        
        if self.training:
            #radius = radius
            phai = phai + 90  # 座標系変換
            self.reset_valid_ptn()
        else:  # 評価モードは固定入力
            if self.valid_ptn % 2 == 0:  # 偶数
                ptn_sign = 1
            else:
                ptn_sign = -1
            radius = 5.5 # 半径5.5mの円周
            phai = 90 + ptn_sign * self.valid_ptn * 10  # 90 ± 10 * valid_ptn (90, 80, 110, 60, 130, 40, 150[deg]を想定)
            self.valid_ptn = self.valid_ptn + 1

        x = -radius * np.sin( np.radians(phai))
        y = -radius * np.cos( np.radians(phai))
        theta = np.radians(90 - phai)  # ゴール方向
        
        self.simulator.reset(init_pose=[x, y, theta], init_ref=[0, 0])
        self.close()
        
        return self.get_observation()
        
    def step(self, action):
        self.simulator.update(action)
        observation = self.get_observation()
        self.reward = self.get_reward(observation)
        
        self.step_cnt = self.step_cnt + 1  # ステップ数カウント
        self.total_reward = self.total_reward + self.reward
        
        if self.training:
            done, self.done_index = self.get_done()
        else:  # 評価モードはdone判定をイベント参加者間で統一
            done, self.done_index = self.get_done_validation()
        
        # 獲得総報酬、doneの理由、ステップ数
        info = {'total_reward': self.total_reward, \
                'done_idx': self.done_index, \
                'step_cnt': self.step_cnt}
        
        return observation, self.reward, done, info

    def render(self, mode='human', text_infos = {}):
        if self.viewer == None:
            self.viewer = Viewer()
            
        return self.viewer.render(self.simulator, text_infos, mode)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_observation(self):
        observation = self.simulator.get_observation()
        return observation["rel_pos"] + observation["ref"]
        
    def get_reward(self, observation):
        # 継承先で定義していないとエラーを出力
        raise NotImplementedError
        
        reward = 0
        
        return reward

    def get_done(self, step_over_thr=600, minus_reward_thr=-50, x_goal_thr=0.2, y_goal_thr=0.1, theta_deg_goal_thr=2):
    # エピソード終了条件判定＆終了原因出力
        done = False
        done_index = 0  # 0:doneしてない、1：ゴール、2：衝突、3:ステップ数超過、4:負の報酬超過
        if self.simulator.get_goaled(x_goal_thr, y_goal_thr, theta_deg_goal_thr):
            done = True
            done_index = 1
        elif self.simulator.get_collision():
            done = True
            done_index = 2
        elif self.step_cnt > step_over_thr:
            done = True
            done_index = 3
        elif self.total_reward < minus_reward_thr:
            done = True
            done_index = 4
        
        return done, done_index

    def get_done_validation(self):
    # 評価用のget_done
    # 報酬や終了ステップ数の設定差による不公平をなくす
        done = False
        done_index = 0  # 0:doneしてない、1：ゴール、2：衝突、3:ステップ数超過
        
        # ゴールしきい値は固定: x方向0.2[m], y方向0.1[m], 回転方向2[deg]
        if self.simulator.get_goaled(x_goal_thr=0.2, y_goal_thr=0.1, theta_deg_goal_thr=2):
            done = True
            done_index = 1
        elif self.simulator.get_collision():
            done = True
            done_index = 2
        elif self.step_cnt > 600:  # 終了ステップ数は固定：600
            done = True
            done_index = 3
        
        return done, done_index
        
    def set_validation(self):
    # 評価モードへの変更
        self.training = False
        
    def set_training(self):
    # 学習モードへの変更
        self.training = True
        
    def reset_valid_ptn(self):
    # 評価モードパターン番号クリア
        self.valid_ptn = 0
        