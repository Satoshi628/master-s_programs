import gym
import numpy as np
from gym import spaces
from envs.Viewer  import Viewer
from envs.PySim2D import PySim2D
from envs.myenv_base import MyEnv_base

class MyEnv(MyEnv_base):
    # __init___ 環境生成時に実行
    def __init___(self):
        ## 報酬設定用
        self.goal_dist_old = 100  # 報酬設定用：近づいた/離れた判定用のメモリ
        ## 親クラスのメソッド呼び出し
        super().__init__(env)
        
    # reset エピソード開始時に実行
    def reset(self):
        ## 報酬設定用
        self.goal_dist_old = 100  # 近づいた/離れた判定用のメモリ
        
        ## 学習初期位置の設定用
        radius=5.5  # 半径5.5mの円
        phai=np.random.randint(low=-60, high=60)  # -60deg ～60degの整数値
        
        ## 親クラスのメソッド呼び出し
        observation = super().reset(radius, phai)
        return observation
        
    # get_reward ここの実装がメイン課題！
    def get_reward(self, observation):
        pos_x = observation[0]       # 相対x位置[m]
        pos_y = observation[1]       # 相対y位置[m]
        pos_theta = observation[2]   # 相対向き[rad]
        speed = observation[3]       # 速さ[m/s]
        curvature = observation[4]   # 曲率[1/m]
        
        reward = 0
        goal_dist = (pos_x ** 2) + (pos_y ** 2)

        
        ## ゴール
        if self.simulator.get_goaled():
            reward = reward + 500
            
        ## 衝突
        if self.simulator.get_collision():
            reward = reward - 500
        
        rad_weight = 0.1
        speed_weight = 0.1

        reward = reward - rad_weight * np.abs(pos_theta) / 2 * np.pi * 360
        reward = reward + speed_weight * speed
        
        ## ゴールに近づいたら正の報酬、離れたら負の報酬
        if self.goal_dist_old > goal_dist:
            reward = reward + 1
        else:
            reward = reward - 5
        
        # 近づいた/離れた判定用のメモリ
        self.goal_dist_old = goal_dist
        
        return reward
    
    # get_done 2つのしきい値を変更可能！
    def get_done(self):
        step_over_thr    = 400  # ゴール失敗として終了するシミュレーションステップしきい値
        minus_reward_thr = -50  # ゴール失敗として終了する負の累積報酬しきい値
        
        ## 親クラスのメソッド呼び出し
        done, done_index = super().get_done(step_over_thr, minus_reward_thr)
        
        return done, done_index
    
    # render アニメーションにテキスト情報の追加が可能（任意）
    def render(self, mode='human'):
        text_infos = {"ptn_no": (self.valid_ptn - 1),
                      "step_cnt": self.step_cnt,
                      "reward": self.reward,
                      "total_reward": self.total_reward
                      }
                      
        frame = super().render(mode, text_infos)
        return frame
    
    # step() ... aigymの枠組みで必須のメソッド.親クラスで実装済み
    #def step(self, action):
    
    # close() ... aigymの枠組みで必須のメソッド.親クラスで実装済み
    #def close(self):
