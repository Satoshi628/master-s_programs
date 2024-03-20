import numpy as np
import matplotlib.pyplot as plt
from rl.callbacks import Callback

index_dict = {'goaled':(1, 'cyan'), 'collision':(2, 'red'), 'step_over':(3, 'orange'), 'minus_reward':(4, 'gray')}
values_list = list(index_dict.values())  # done_index辞書型のvalueをlistに変換
key_list = list(index_dict.keys())       # done_index辞書型のkeyをlistに変換

class TrainLogger(Callback):
# 学習の集計
    def __init__(self):
        self.step_cnt = []      # 実行ステップ数
        self.done_index = []    # エピソード終了理由
        self.total_reward = []  # エピソード総報酬
        self.metrics = []
        
        # 学習ログ用データ
        self.last_episode_num = 0  # 最終エピソード番号
        self.interval_episode_num = [0]  # エピソード区間の番号
        self.sum_list = [[] for i in range(len(values_list))]  # [[], [], [], []] 各done_indexごとの発生回数
        
        self.episode_reward = []  # 学習ループエピソード報酬
        
    def on_step_end(self, step, logs):
        self.metrics.append(logs['metrics'])
        
    def on_episode_end(self, episode, logs):
        # エピソード毎の集計
        self.step_cnt.append(self.env.step_cnt)
        self.done_index.append(self.env.done_index)
        self.total_reward.append(self.env.total_reward)
        
        self.last_episode_num = episode
        
    def on_train_end(self, logs):
        # 学習インターバル毎の集計
        temp = self.interval_episode_num[-1] + self.last_episode_num + 1
        self.interval_episode_num.append(temp)
        
        # 呼び出した時点のループインターバル区間のdone_indexを集計する
        pre_interval_episode_num = self.interval_episode_num[-2]
        now_interval_episode_num = self.interval_episode_num[-1]
        for i in range(len(values_list)):
            dev_list = [1 if idx==values_list[i][0] else 0  for idx in self.done_index[pre_interval_episode_num : now_interval_episode_num] ]
            self.sum_list[i].append(sum(dev_list))
        
        # 学習エピソードごとの総報酬
        self.episode_reward.append(sum(self.total_reward[pre_interval_episode_num : now_interval_episode_num])/(now_interval_episode_num - pre_interval_episode_num))
    
# 評価エピソードの集計
class TestLogger(Callback):
    def __init__(self):
        # done_indexの定義
        self.step_cnt = []      # 実行ステップ数
        self.done_index = []    # エピソード終了理由
        self.total_reward = []  # エピソード総報酬
        
        # 評価ログ用データ
        self.total_takt_time_list = []
        self.total_takt_time_list_raw = []  # ペナルティなし
        self.goal_rate_list = []
        
        self.sum_list = [[] for i in range(len(values_list))]  # [[], [], [], []] 各done_indexごとの発生回数
        
    def on_episode_end(self, episode, logs):
        # エピソード毎の集計
        self.step_cnt.append(self.env.step_cnt)
        self.done_index.append(self.env.done_index)
        self.total_reward.append(self.env.total_reward)
        
    def calc_test_result(self, nb_valid_episodes):
        # 評価結果の集計
        # ゴール以外は60sとして集計。ステップ時間0.1はマジックナンバーなので注意
        takt_times_list = [cnt*0.1 if done_index == index_dict['goaled'][0] else 60 \
                           for cnt, done_index in zip(self.step_cnt, self.done_index )]
        takt_times_list_raw = [cnt*0.1 for cnt in self.step_cnt]
        
        total_takt_time = sum(takt_times_list)
        total_takt_time_raw = sum(takt_times_list_raw)
        self.total_takt_time_list.append(total_takt_time)
        self.total_takt_time_list_raw.append(total_takt_time_raw)
        
        ## ゴール率表示
        nb_goal = self.done_index.count(index_dict['goaled'][0])  # ゴールした回数をカウント
        self.goal_rate = nb_goal / nb_valid_episodes * 100
        self.goal_rate_list.append(self.goal_rate)
        
        ## 終了条件の積み上げ
        for i in range(len(values_list)):
            dev_list = [1 if idx==values_list[i][0] else 0  for idx in self.done_index ]
            self.sum_list[i].append(sum(dev_list))
        
    def reset(self):
        # リセット
        self.step_cnt = []      # 実行ステップ数
        self.done_index = []    # エピソード終了理由
        self.total_reward = []  # エピソード総報酬
    
# 2Dアニメーション生成用コールバック
class AnimationLogger(Callback):
    def __init__(self):
        self.frames = []
        
    def on_step_end(self, step, logs={}):
        self.frames.append(self.env.render(mode="rgb_array"))
        
    def save_animation(self, filename):
        import cv2
        out = None
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_path = "./results/" + filename + ".wmv"
        for i in range(len(self.frames)):
            im = self.frames[i]
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            if out is None:
                w = im.shape[1]
                h = im.shape[0]
                out = cv2.VideoWriter(video_path, fourcc, 10, (w, h))  # 0.1s->10Hz
            out.write(im)
        out.release()
        
# csv生成用コールバック
class CsvLogger(Callback):
    def __init__(self):
        self.rows = []  # [[Name, time, x, y, z, roll, pitch, yaw ], [...]...]
        self.rows.append(['Name', 'time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'speed', 'curvature'])
        
    def on_step_end(self, step, logs={}):
        observation = self.env.get_observation()
        # Name, time, x, y, z, roll, pitch, yaw
        step_cnt = self.env.step_cnt
        Name = "ptn_no_"+str(self.env.valid_ptn)+"_"+str(step_cnt).zfill(3)
        self.rows.append([Name, round(step_cnt*0.1, 1 ), \
                          observation[0], observation[1], 0,\
                          0, 0, observation[2], observation[3], observation[4]])
        
    def save_csv(self, filename):
        import csv
        csv_path = "./results/" + filename + ".csv"
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.rows)
            
    
# 学習ログ進捗のプロット
def plot_train_logger(cb_train_logger, nb_loops):
    fig1 = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig1.add_subplot(1, 1, 1)
    h1 = []
    l1 = key_list
    for i in range(len(cb_train_logger.sum_list[0])):   # i : 取得したインターバルの回数
        bottom = 0
        for j in range(len(cb_train_logger.sum_list)):  # j : 取得したデータ種類の数 (goaled, collision, ...)
            data = cb_train_logger.sum_list[j][i]
            line, = ax1.bar(i, data, bottom = bottom, color = values_list[j][1])
            bottom = bottom + data
            h1.append(line)
    ax1.set_xlim([-0.5, nb_loops -0.5 ])  # 両端の表示切れを防ぐため、上下限それぞれ0.5ずらしている
    ax1.set_xticks(np.arange(0, nb_loops, 1), minor=False)
    ax1.set_ylabel("done index count[-]")
    ax1.set_xlabel('loop_nb')
    
    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, len(cb_train_logger.episode_reward), 1), cb_train_logger.episode_reward,\
             marker='.', color='black', label='episode_reward')
    ax2.set_ylabel("episode reward[-]")
    
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1[0:len(l1)] + h2, l1 + l2, loc='upper left', bbox_to_anchor=(1.2, 1))
    ax1.set_title('train processing')
    
    # グラフ表示
    plt.show()
    
    return fig1

# 評価ログ進捗のプロット
def plot_test_logger(cb_test_logger, nb_loops):
    fig1 = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig1.add_subplot(1, 1, 1)
    h1 = []
    l1 = key_list
    for i in range(len(cb_test_logger.sum_list[0])):   # i : 取得したインターバルの回数
        bottom = 0
        for j in range(len(cb_test_logger.sum_list)):  # j : 取得したデータ種類の数 (goaled, collision, ...)
            data = cb_test_logger.sum_list[j][i]
            line, = ax1.bar(i, data, bottom = bottom, color = values_list[j][1])
            bottom = bottom + data
            h1.append(line)
    ax1.set_xlim([-0.5, nb_loops -0.5 ])  # 両端の表示切れを防ぐため、上下限それぞれ0.5ずらしている
    ax1.set_xticks(np.arange(0, nb_loops, 1), minor=False)
    ax1.set_ylabel("done index count[-]")
    ax1.set_xlabel('loop_nb')
    
    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, len(cb_test_logger.total_takt_time_list), 1), cb_test_logger.total_takt_time_list,\
                       marker='.', color='black', label='total_takt_time(penalty)')
    ax2.plot(np.arange(0, len(cb_test_logger.total_takt_time_list_raw), 1), cb_test_logger.total_takt_time_list_raw,\
                       marker='.', color='gray', label='total_takt_time')
    ax2.set_ylim([0, 60*7+1])  # penaltyを考慮した最大値60*7
    ax2.set_ylabel("total takt time[s]")
    
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1[0:len(l1)] + h2, l1 + l2, loc='upper left', bbox_to_anchor=(1.2, 1))
    ax1.set_title('test processing')
    
    # グラフ表示
    plt.show()
    
    return fig1

# レポート用のプロット
def plot_report(cb_test_logger, trial_name="try_xxx", weights_no="yy"):
    cb_test_logger.step_cnt
    fig1 = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig1.add_subplot(1, 1, 1)
    
    nb_validation = len(cb_test_logger.step_cnt)
    
    for i in range(nb_validation):
        takt_time = cb_test_logger.step_cnt[i] * 0.1
        if cb_test_logger.done_index[i] == index_dict['goaled'][0]:  # ゴールした
            ax1.bar(i, takt_time, color='cyan')  # 1step あたり0.1s
        else:
            color = 'red'
            ax1.bar(i, takt_time, color='red')  # 1step あたり0.1s
            ax1.bar(i, 60 - takt_time, bottom = takt_time, color='gray', alpha=0.5)  # ペナルティ分
        
    fig1.text(0.9, 0.8, "total takt time="+ '{:.1f}'.format(cb_test_logger.total_takt_time_list[0]) +"[s]",\
              ha = 'right', size=20, color = "black")
    fig1.text(0.9, 0.7, "goal rate="+ '{:.1f}'.format(cb_test_logger.goal_rate) +"[%]",\
              ha = 'right', size=20, color = "black")
    
    ax1.set_xticks(np.arange(0, nb_validation, 1), minor=False)
    ax1.set_xlabel('validation nb')
    ax1.set_ylabel("takt time[s]")
    ax1.set_ylim([0, 61])  # 最大値60sまで表示
    ax1.set_title(trial_name+'_'+weights_no)
    
    # グラフ表示
    plt.show()
    
    return fig1
    
# metricの表示
def plot_metrics(agent, cb_train_logger, mv_num):
    data_len = len(cb_train_logger.metrics)
    loss = np.array(cb_train_logger.metrics)[:,0]
    metrics_name = agent.metrics_names[0]
    b=np.ones(mv_num)/mv_num
    loss_mv = np.convolve(loss, b, mode='same')  #移動平均

    fig1 = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(np.arange(0, data_len, 1), loss_mv, label=metrics_name)

    ax1.set_xlabel('step')
    ax1.set_ylabel("loss[-]")
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='upper left', bbox_to_anchor=(1, 1))
    ax1.set_title('training metrics')
    
    # グラフ表示
    plt.show()
    return fig1