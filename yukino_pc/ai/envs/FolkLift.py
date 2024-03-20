import numpy as np

#形のある物体
## 座標系
#  画面上がX正方向
#  画面左がY正方向
class FolkLift(object):
    def __init__(self, dt):
        self.wheel_base = 1.50  # ホイールベース[m]
        self.front_len = 1.53   # 前輪から先端までの距離[m]
        self.rear_len  = 0.47   # 後輪から後端までの距離[m]
        self.width  = 1.80      # 車両幅[m]
        self.dt  = dt      # ステップ時間[s]
        
        self._pose = [0, 0, 0]    # 前輪の位置と方向
        self._reference = [0, 0]  # 速さと曲率
    
    #初期パラメータを設定
    def set_init_state(self, init_pose = [0, 0, 0], init_ref=[0, 0]):
        self._pose = init_pose      # 前輪の位置と角度
        self._reference = init_ref  # 直交座標の速度
        
    def get_points(self):  # 4角の位置list [4, 2] xy形式
    # https://teratail.com/questions/134185
    # 0----1
    # |    |
    # 3----2
        pose_x = self._pose[0]
        pose_y = self._pose[1]
        pose_theta = self._pose[2]
        # Untreated_points回転前の4角の位置 np.array(4, 2)
        Untreated_points = np.array([[self.front_len,   self.width/2 ],
                                     [self.front_len, - self.width/2 ],
                                     [-(self.rear_len + self.wheel_base),  - self.width/2 ],
                                     [-(self.rear_len + self.wheel_base),    self.width/2 ]])
        # 回転行列
        R = np.array([[np.cos(pose_theta), -np.sin(pose_theta)],
                      [np.sin(pose_theta),  np.cos(pose_theta)]])
                      
        treated_points = []
        for point_xy in Untreated_points:
            treated_point_xy = np.dot(R, point_xy).tolist()
            treated_points.append(treated_point_xy)  # list型でappendするほうnp.arrayよりが早い
            
        offset_points = np.array([[pose_x, pose_y ],
                                  [pose_x, pose_y ],
                                  [pose_x, pose_y ],
                                  [pose_x, pose_y ]])
        treated_points = np.array(treated_points) + offset_points
        return treated_points.tolist()
        
    #位置を設定
    def set_pose(self, pose):
        self._pose = pose
        
    #位置を取得
    def get_pose(self):
        return self._pose
        
    #フォーク先端の位置を取得
    def get_front_pose(self):
        f_pose_theta = self._pose[2]
        f_pose_x = self._pose[0] + self.front_len * np.cos(self._pose[2])
        f_pose_y = self._pose[1] + self.front_len * np.sin(self._pose[2])
        
        return [f_pose_x, f_pose_y, f_pose_theta ]
    
    #速度を設定
    def set_reference(self, reference):
        self._reference = reference

    #速度を設定
    def get_reference(self):
        return self._reference

    #移動
    def move(self):
        dx     = self._reference[0] * np.cos(self._pose[2]) * self.dt
        dy     = self._reference[0] * np.sin(self._pose[2]) * self.dt
        dThata = self._reference[0] * self._reference[1] * self.dt
        self.set_pose(self._pose + np.array([dx, dy, dThata]))
    
