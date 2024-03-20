import numpy as np
from scipy import signal
from scipy import fftpack
import soundfile as sf
from matplotlib import pyplot as plt

from moviepy.editor import AudioFileClip

# 動画ファイルの指定
video_file = "test.mp4"

# 動画からAudioFileClipオブジェクトを生成
audio = AudioFileClip(video_file)

# .wavファイルとして保存
audio.write_audiofile(video_file.replace('.mp4', '.wav'))


def wavload(path):
    data, samplerate = sf.read(path)
    return data, samplerate

def ov(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate                     # 全データ長
    Fc = Fs / samplerate                            # フレーム周期
    x_ol = Fs * (1 - (overlap / 100))               # オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap / 100))) /\
                (Fc * (1 - (overlap / 100))))       # 抽出するフレーム数（平均化に使うデータ個数）

    array = []                                      # 抽出したデータを入れる空配列の定義

    # forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)                          # 切り出し位置をループ毎に更新
        array.append(data[ps:ps + Fs:1])            # 切り出し位置psからフレームサイズ分抽出して配列に追加
        final_time = (ps + Fs)/samplerate           #切り出したデータの最終時刻
    return array, N_ave, final_time                 # オーバーラップ抽出されたデータ配列とデータ個数、最終時間を戻り値にする

def hanning(data_array, Fs, N_ave):
    han = signal.hann(Fs)                           # ハニング窓作成
    acf = 1 / (sum(han) / Fs)                       # 振幅補正係数(Amplitude Correction Factor)

    # オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * han         # 窓関数をかける

    return data_array, acf

def db(x, dBref):
    y = 20 * np.log10(x / dBref)                   # 変換式
    return y                                       # dB値を返す

def aweightings(f):
    if f[0] == 0:
        f[0] = 1
    else:
        pass
    ra = (np.power(12194, 2) * np.power(f, 4)) / \
            ((np.power(f, 2) + np.power(20.6, 2)) * \
            np.sqrt((np.power(f, 2) + np.power(107.7, 2)) * \
                    (np.power(f, 2) + np.power(737.9, 2))) * \
            (np.power(f, 2) + np.power(12194, 2)))
    a = 20 * np.log10(ra) + 2.00
    return a

def fft_ave(data_array, samplerate, Fs, N_ave, acf):
    fft_array = []
    fft_axis = np.linspace(0, samplerate, Fs)      # 周波数軸を作成
    a_scale = aweightings(fft_axis)                # 聴感補正曲線を計算

    # FFTをして配列にdBで追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
    for i in range(N_ave):
        fft_array.append(db\
                        (acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2))\
                        , 2e-5))

    fft_array = np.array(fft_array) + a_scale      # 型をndarrayに変換しA特性をかける
    fft_mean = np.mean(fft_array, axis=0)          # 全てのFFT波形の平均を計算

    return fft_array, fft_mean, fft_axis


path = "test.wav"                           #ファイルパスを指定
data, samplerate = wavload(path)   #wavファイルを読み込む
data = data[:,0]
x = np.arange(0, len(data)) / samplerate    #波形生成のための時間軸の作成

# Fsとoverlapでスペクトログラムの分解能を調整する。
Fs = 4096                                   # フレームサイズ
overlap = 90                                # オーバーラップ率

# オーバーラップ抽出された時間波形配列
time_array, N_ave, final_time = ov(data, samplerate, Fs, overlap)

# ハニング窓関数をかける
time_array, acf = hanning(time_array, Fs, N_ave)

# FFTをかける
fft_array, fft_mean, fft_axis = fft_ave(time_array, samplerate, Fs, N_ave, acf)

# スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
fft_array = fft_array.T

# ここからグラフ描画
# グラフをオブジェクト指向で作成する。
fig = plt.figure()
ax1 = fig.add_subplot(111)
print(fft_array.shape)

# データをプロットする。
im = ax1.imshow(fft_array, \
                vmin = 0, vmax = 60,
                extent = [0, final_time, 0, samplerate], \
                aspect = 'auto',\
                cmap = 'jet')

# カラーバーを設定する。
cbar = fig.colorbar(im)
cbar.set_label('SPL [dBA]')

# 軸設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')

# スケールの設定をする。
ax1.set_xticks(np.arange(0, 50, 5))
ax1.set_yticks(np.arange(0, 20000, 2000))
ax1.set_xlim(0, 60*2)
ax1.set_ylim(0, 10000)

# グラフを表示する。
plt.show()
plt.close()