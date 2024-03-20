import csv
import numpy as np
from .FolkLift import FolkLift

## 役割：強化学習イベント参加者が基本的に触らない部分のコード
# 座標系：画面上がX正方向,画面左がY正方向
# delta_t定義
# シミュレータ更新
# 衝突判定
# ゴール条件

class PySim2D(object):
    def __init__(self, file_path):
        self.dt = 0.1  # シミュレーションステップ時間[s]
        self.goal = [0, 0, 0]  # ゴール位置
        self.FolkLift = FolkLift(self.dt)
        
        # csvから衝突領域を読み込み。(x,y, x,y)形式で領域を指定
        self._collision_areas = []
        with open(file_path) as f:
            reader = csv.reader(f)
            l_str = [row for row in reader]
            self._collision_areas = [[int(v) for v in row] for row in l_str]
            
        # 速さ(方向を含まない)[m/s]： speed
        self.max_spd = 0.5  # 速さ指令値の最大値。正と負で絶対値が同じ
        self.spd_step_num = 3  # 速さ指令値の離散化ステップ数。速度0をとるため奇数(整数型.浮動の演算誤差防止)
        self.delta_spd = (2 * self.max_spd)/(self.spd_step_num -1)  # 1ステップ当たりの速さ指令値の変化
        self.int_spd = 0  # 速さ指令値の離散値 (整数型.浮動の演算誤差防止)
        
        # 曲率[1/m]： curvature
        self.curv_step_num = 9  # 曲率の離散化ステップ数。速度0をとるため奇数(整数型.浮動の演算誤差防止)
        self.delta_curv = 0.35  # 1ステップ当たりの曲率の変化.車体の向きの変化量のLSBが1degとなる値 
        self.int_curv = 0  # 曲率の離散値 (整数型.浮動の演算誤差防止)
        
    # 初期化
    def reset(self, init_pose=[0,0,0], init_ref=[0,0]):
        # 注意!現在のコードだとinit_refが非ゼロの値の場合には対応していない。（int_*の初期化のため）
        self.int_spd  = 0
        self.int_curv = 0
        self.FolkLift.set_init_state(init_pose, init_ref)  # FolkLiftを初期化
        
    # シミュレーションを次の状態にする
    def update(self, action):
    # actionの変換
    # 変換前 action：0～8 速さと曲率の加減の組み合わせ
    # 変換後 speed, curvature：速さと曲率
        ## actionの適用
        # spd_action :速度加減(0:そのまま, 1:増, 2:減,)
        # curv_action:曲率加減(0:そのまま, 1:増, 2:減)
        
        spd_action  = int(action / 3)  # 速さの加減
        curv_action = int(action % 3)  # 曲率の加減
        ## 速度[m/s]
        if spd_action == 0:
            spd_acc = 0
        elif spd_action == 1:
            spd_acc = 1
        elif spd_action == 2:
            spd_acc = -1
        else:
            spd_acc = 0
        self.int_spd = self.int_spd + spd_acc
        
        # 離散化した速度.上下限処理
        int_spd_max = (self.spd_step_num - 1)/2
        if self.int_spd > int_spd_max:
            self.int_spd = int_spd_max
        elif self.int_spd < -int_spd_max:
            self.int_spd = -int_spd_max
        
        speed = self.int_spd * self.delta_spd
        
        ## 曲率[1/m]
        if curv_action == 0:
            curv_acc = 0
        elif curv_action == 1:
            curv_acc = 1
        elif curv_action == 2:
            curv_acc = -1
        else:
            curv_acc = 0
        self.int_curv = self.int_curv + curv_acc

        # 離散化した曲率.上下限処理
        int_curv_max = (self.curv_step_num - 1)/2
        if self.int_curv > int_curv_max:
            self.int_curv = int_curv_max
        elif self.int_curv < -int_curv_max:
            self.int_curv = -int_curv_max
            
        curvature = self.int_curv * self.delta_curv
        
        # フォークリフトの移動量を計算
        self.FolkLift.set_reference([speed, curvature])
        self.FolkLift.move()
        
    ## observation
    def get_observation(self):
        # pose:車両の絶対位置(x, y, theta)
        # ref :車両が受信した指令値(speed, curvature)
        # rel_pos:車両とパレットの相対位置(x', y', theta')
        observation = {}
        observation["pose"] = self.FolkLift.get_pose()
        observation["ref"] = self.FolkLift.get_reference()
        
        observation["rel_pos"] = [0,0,0]  # ゴール基準の相対位置初期化
        [f_pose_x, f_pose_y, f_pose_theta] = self.FolkLift.get_front_pose()
        observation["rel_pos"][0] = self.goal[0] - f_pose_x
        observation["rel_pos"][1] = self.goal[1] - f_pose_y
        observation["rel_pos"][2] = self.goal[2] - f_pose_theta
        
        return observation
        
    ## ゴール判定
    def get_goaled(self, x_goal_thr=0.2, y_goal_thr=0.1, theta_deg_goal_thr=2):
        [f_pose_x, f_pose_y, f_pose_theta] = self.FolkLift.get_front_pose()
        
        if   (self.goal[0] - f_pose_x > 0)\
          and(self.goal[0] - f_pose_x < x_goal_thr)\
          and(np.fabs(self.goal[1] - f_pose_y) < y_goal_thr)\
          and(np.fabs(self.goal[2] - f_pose_theta) < np.radians(theta_deg_goal_thr)) :
            return True
        else:
            return False
        
    ## 衝突領域侵入判定
    def get_collision(self):
        # 線分ab, cdが交差する場合True
        # 端点が他方の線分上にある場合もTrue 端点が他方の線分の延長線上にある場合もTrueを返すので注意
        # https://qiita.com/zu_rin/items/e04fdec4e3dec6072104
        def _JudgeCollision( a, b, c, d):
            s = (a[0] - b[0]) * (c[1] - a[1]) - (a[1] - b[1]) * (c[0] - a[0])
            t = (a[0] - b[0]) * (d[1] - a[1]) - (a[1] - b[1]) * (d[0] - a[0])
            if (s * t > 0):
                return False
            s = (c[0] - d[0]) * (a[1] - c[1]) - (c[1] - d[1]) * (a[0] - c[0])
            t = (c[0] - d[0]) * (b[1] - c[1]) - (c[1] - d[1]) * (b[0] - c[0])
            if (s * t > 0):
                return False
            return True
            
        # どこか1か所でも衝突領域に侵入していれば即座にFalseを返す（探すのをやめる）
        points = self.FolkLift.get_points()
        
        for p_area in self._collision_areas:
            lt_x = p_area[0]  # p_area[0]:左上x位置 lt_x
            lt_y = p_area[1]  # p_area[1]:左上y位置 lt_y
            rb_x = p_area[2]  # p_area[2]:右下x位置 rb_x
            rb_y = p_area[3]  # p_area[3]:右下y位置 rb_y
            for i in range(len(points)):
                # 衝突領域の判定処理
                rlst = _JudgeCollision([lt_x, lt_y], [rb_x, lt_y],\
                                       points[i-1],  points[i] )
                if rlst:
                    return True
                    
                rlst = _JudgeCollision([rb_x, lt_y], [rb_x, rb_y],\
                                       points[i-1],  points[i] )
                if rlst:
                    return True
                    
                rlst = _JudgeCollision([rb_x, rb_y], [lt_x, rb_y],\
                                       points[i-1],  points[i] )
                if rlst:
                    return True
                    
                rlst = _JudgeCollision([lt_x, rb_y], [lt_x, lt_y],\
                                       points[i-1],  points[i] )
                if rlst:
                    return True
        return False
        
