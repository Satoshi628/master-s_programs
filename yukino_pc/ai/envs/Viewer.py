import numpy as np
import envs.rendering as rendering
import pyglet

class DrawText:
# https://stackoverflow.com/questions/56744840/pyglet-label-not-showing-on-screen-on-draw-with-openai-gym-render
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()

class Viewer(object):
    def __init__(self):
        self._screen_width = 500
        self._screen_height = 500
        self._viewer = rendering.Viewer(self._screen_width, self._screen_height)
        self._viewer.set_bounds(-100, 100, -150, 50)  # (left, right, bottom, top)
        self._create = False
        self._fl_trans = []
        self._Lfork_trans = []
        self._Rfork_trans = []
        self._Flwheel_trans = []
        self._Frwheel_trans = []
        #self._Rwheel_trans = []
        self._Rlwheel_trans = []
        self._Rrwheel_trans = []
        self._path_trans = []
        self._scale = 10  # 表示スケール倍率

    # レンダリング
    def render(self, simulator, text_infos, mode):
        fl_pose = simulator.FolkLift.get_pose()
        w_base = simulator.FolkLift.wheel_base
        # 描画物体は初回のみ生成
        if self._create == False:
            self._create = True
            
            # フォークリフトの描画と更新用データの定義
            fl_points = simulator.FolkLift.get_points()  # points:左上から四つ角の(x, y)
            self._fl_trans = rendering.Transform()
            # 運動学平面のlt(x, y), rb(x, y)を算出する
            fork_length = ( 3.5 - 0.5 )
            fork_width  = 1.80/2.0
            t_x = fl_pose[0] + 0.5
            l_y = fl_pose[1] + fork_width
            b_x = fl_pose[0] - fork_length+1
            r_y = fl_pose[1] - fork_width
            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
                                             [fl_pose[0], fl_pose[1], 0 ])
            geom.set_color(1.0 ,153/255, 0.0)  # RGB 薄いオレンジ
            geom.add_attr(self._fl_trans)
            self._viewer.add_geom(geom)
            
            # フォーク(左)
            self._Lfork_trans = rendering.Transform()
            fork_length = 0.6
            fork_width  = 0.15
            t_x = fl_pose[0] + fork_length + 1.
            l_y = fl_pose[1] + fork_width + 0.5
            b_x = fl_pose[0] - fork_length + 1.
            r_y = fl_pose[1] - fork_width + 0.5
            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
                                             [fl_pose[0], fl_pose[1], 0 ])
            geom.set_color(192/255 ,192/255, 192/255)  # RGB 灰色
            geom.add_attr(self._Lfork_trans)
            self._viewer.add_geom(geom)

            # フォーク(右)
            self._Rfork_trans = rendering.Transform()
            fork_length = 0.6
            fork_width  = 0.15
            t_x = fl_pose[0] + fork_length + 1.
            l_y = fl_pose[1] + fork_width - 0.5
            b_x = fl_pose[0] - fork_length + 1.
            r_y = fl_pose[1] - fork_width - 0.5
            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
                                             [fl_pose[0], fl_pose[1], 0 ])
            geom.set_color(192/255 ,192/255, 192/255)  # RGB 灰色
            geom.add_attr(self._Rfork_trans)
            self._viewer.add_geom(geom)

            # 前輪(左)
            self._Flwheel_trans = rendering.Transform()
            tire_length = 0.4
            tire_width  = 0.2
            t_x = fl_pose[0] + tire_length
            l_y = fl_pose[1] + tire_width + 0.8
            b_x = fl_pose[0] - tire_length
            r_y = fl_pose[1] - tire_width + 0.8
            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
                                             [fl_pose[0], fl_pose[1], 0 ])
            geom.set_color(0.0 ,51/255, 102/255)  # RGB 濃い青
            geom.add_attr(self._Flwheel_trans)
            self._viewer.add_geom(geom)
            
            # 前輪(右)
            self._Frwheel_trans = rendering.Transform()
            tire_length = 0.4
            tire_width  = 0.2
            t_x = fl_pose[0] + tire_length
            l_y = fl_pose[1] + tire_width - 0.8
            b_x = fl_pose[0] - tire_length
            r_y = fl_pose[1] - tire_width - 0.8
            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
                                             [fl_pose[0], fl_pose[1], 0 ])
            geom.set_color(0.0 ,51/255, 102/255)  # RGB 濃い青
            geom.add_attr(self._Frwheel_trans)
            self._viewer.add_geom(geom)
            
            # 後輪
#            self._Rwheel_trans = rendering.Transform()
#            t_x = tire_length
#            l_y = tire_width
#            b_x = - tire_length
#            r_y = - tire_width
#            
#            tire_ang = simulator.get_observation()["ref"][1]
#            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
#                                             [0, 0, 0 ])
#            geom.set_color(0.0 ,51/255, 102/255)  # RGB 濃い青
#            geom.add_attr(self._Rwheel_trans)
#            self._viewer.add_geom(geom)

            # 後輪(左)
            self._Rlwheel_trans = rendering.Transform()
            t_x = tire_length
            l_y = tire_width
            b_x = - tire_length
            r_y = - tire_width
            
            tire_ang = simulator.get_observation()["ref"][1]
            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
                                             [0, 0, 0 ])
            geom.set_color(0.0 ,51/255, 102/255)  # RGB 濃い青
            geom.add_attr(self._Rlwheel_trans)
            self._viewer.add_geom(geom)

            # 後輪(右)
            self._Rrwheel_trans = rendering.Transform()
            t_x = tire_length
            l_y = tire_width
            b_x = - tire_length
            r_y = - tire_width
            
            tire_ang = simulator.get_observation()["ref"][1]
            geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]],\
                                             [0, 0, 0 ])
            geom.set_color(0.0 ,51/255, 102/255)  # RGB 濃い青
            geom.add_attr(self._Rrwheel_trans)
            self._viewer.add_geom(geom)

            # 禁止領域の描画と更新用データの定義
            p_areas_points = simulator._collision_areas
            for p_area_points in p_areas_points:
                # 運動学平面のltrbを算出する
                t_x = p_area_points[0]
                l_y = p_area_points[1]
                b_x = p_area_points[2]
                r_y = p_area_points[3]
                
                geom = self._create_polygon_geom([[t_x, l_y], [t_x, r_y], [b_x, r_y], [b_x, l_y]] )
                geom.set_color(151/255 ,51/255, 0)  # RGB 茶色
                self._viewer.add_geom(geom)
            
        ## オブジェクトの移動更新
        # viewerでは左上を原点とするため、xy入れ替え,回転は逆になる。
        # 車体
        self._fl_trans.set_translation(-fl_pose[1] * self._scale, fl_pose[0] * self._scale)
        self._fl_trans.set_rotation(fl_pose[2])
        
        # フォーク(左)
        self._Lfork_trans.set_translation(-fl_pose[1] * self._scale, fl_pose[0] * self._scale)
        self._Lfork_trans.set_rotation(fl_pose[2])
        
        # フォーク(右)
        self._Rfork_trans.set_translation(-fl_pose[1] * self._scale, fl_pose[0] * self._scale)
        self._Rfork_trans.set_rotation(fl_pose[2])
        
        # 前輪(左)
        self._Flwheel_trans.set_translation(-fl_pose[1] * self._scale, fl_pose[0] * self._scale)
        self._Flwheel_trans.set_rotation(fl_pose[2])
        
        # 前輪(右)
        self._Frwheel_trans.set_translation(-fl_pose[1] * self._scale, fl_pose[0] * self._scale)
        self._Frwheel_trans.set_rotation(fl_pose[2])
        
        # 後輪
#        tire_ang = np.arctan2( -w_base * simulator.get_observation()["ref"][1], 1 )
#        self._Rwheel_trans.set_translation((-fl_pose[1] + w_base*np.sin(fl_pose[2])) * self._scale,\
#                                           (fl_pose[0] - w_base*np.cos(fl_pose[2])) * self._scale)
#        self._Rwheel_trans.set_rotation(fl_pose[2] + tire_ang )

        # 後輪(左)
        tire_ang = np.arctan2( w_base * simulator.get_observation()["ref"][1], 1 )
        self._Rlwheel_trans.set_translation((-fl_pose[1] + w_base*np.sin(fl_pose[2]) + 0.8*np.cos(fl_pose[2])) * self._scale,\
                                           (fl_pose[0] - w_base*np.cos(fl_pose[2]) + 0.8*np.sin(fl_pose[2])) * self._scale)
        self._Rlwheel_trans.set_rotation(fl_pose[2] + tire_ang )

        # 後輪(右)
        tire_ang = np.arctan2( w_base * simulator.get_observation()["ref"][1], 1 )
        self._Rrwheel_trans.set_translation((-fl_pose[1] + w_base*np.sin(fl_pose[2]) - 0.8*np.cos(fl_pose[2])) * self._scale,\
                                           (fl_pose[0] - w_base*np.cos(fl_pose[2]) - 0.8*np.sin(fl_pose[2])) * self._scale)
        self._Rrwheel_trans.set_rotation(fl_pose[2] + tire_ang )
        
        # 軌跡
        self._path_trans = rendering.Transform()
        if np.fabs(tire_ang) > 0.001:  # 円弧軌跡.浮動小数の非ゼロ判定
            path_radius = w_base / np.tan(tire_ang) * self._scale
            path_center = [fl_pose[0] * self._scale + path_radius * np.sin(fl_pose[2]), \
                           fl_pose[1] * self._scale - path_radius * np.cos(fl_pose[2])]
            geom = rendering.make_circle(radius=path_radius, filled=False)
            geom.add_attr(self._path_trans)
            self._path_trans.set_translation(-path_center[1], path_center[0])
            self._viewer.add_onetime(geom)
        
        # 文字列表示
        for cnt, dict_idx in enumerate(text_infos):
            if text_infos[dict_idx] < 0:  # 負の値なら赤色表示
                color=(255, 0, 0, 255)
            else:  # 通常は黒色表示
                color=(0, 0, 0, 255)
            text_infos[dict_idx] = format(text_infos[dict_idx], '.0f') # 小数点以下は表示しない
            info_text = dict_idx + ":" + str(text_infos[dict_idx])
            
            i_label = pyglet.text.Label(
                info_text,
                font_size = 6,
                x = 99,
                y = 45 - 12 * cnt,
                anchor_x="right",
                anchor_y="center",
                color=color,
            )
            i_label.draw()
            self._viewer.add_onetime(DrawText(i_label))
        
        return self._viewer.render(return_rgb_array = mode=='rgb_array')

    #ウィンドウを閉じる
    def close(self):
        if self._viewer:
            self._viewer.close()

    #壁や土台など指定フォーマットで記述された図形の生成
    def _create_polygon_geom(self, ltrb_xy, org_pos = [0, 0, 0]):
        # ltrb_xy = [[lt_x, lt_y], [lb_x, lb_y], [rb_x, rb_y], [rt_x, rt_y] ]
        # org_pos:回転中心
        ## フォークリフトの描画
        # viewerでは左上を原点とするため、元の直交座標のtopがbottomになり、bottomがtopになる。
        geoms = []
        pose_x, pose_y, pose_theta = org_pos
        offset_points = np.array([[pose_x, pose_y ],
                                  [pose_x, pose_y ],
                                  [pose_x, pose_y ],
                                  [pose_x, pose_y ]])
        Untreated_points = np.array(ltrb_xy) - offset_points
        R = np.array([[ np.cos(pose_theta ), np.sin(pose_theta )],
                      [-np.sin(pose_theta ), np.cos(pose_theta )]])
                      
        treated_points = []
        for point_xy in Untreated_points:
            treated_point_xy = np.dot(R, point_xy).tolist()
            treated_points.append(treated_point_xy)  # list型でappendするほうnp.arrayよりが早い
            
        # FilledPolygon:画面左上から時計回り順に座標(y, x) <- 車両運動学座標基準を指定
        g = rendering.FilledPolygon([(treated_points[0][1] * self._scale, treated_points[0][0] * self._scale),\
                                     (treated_points[3][1] * self._scale, treated_points[3][0] * self._scale),\
                                     (treated_points[2][1] * self._scale, treated_points[2][0] * self._scale),\
                                     (treated_points[1][1] * self._scale, treated_points[1][0] * self._scale)])
        
        geoms.append(g)
        
        return rendering.Compound(geoms)
