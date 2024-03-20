#----- Standard Library -----#
import os

#----- Public Package -----#
from tqdm import tqdm
import numpy as np
import cv2
import torch
from PIL import Image
from IPython.display import display

#----- Module -----#
from models.Unet_3D import UNet_3D
from utils.dataset import data_info
from utils.visual import VisualizationTrajectory
from utils.Coord2TrackKalman import Coord2TrackKalman, global_association_solver, TrackInterpolate


video_path = "/mnt/kamiya/dataset/root/data/sample1/2x_sample1.avi"


# GPUモードの設定
if not torch.cuda.is_available():
    print("GPUモードにしてください。")
device = torch.device('cuda:0')

param_dict = {
    "detection_th": 0.4, #検出の閾値(0~1)。高いほど厳しく判定
    "speed_upper1":11.77, #1フレーム間での成長スピードの上限
    "speed_upper2":11.77, #2フレーム間での成長スピードの上限
    "init_speed":[0., 11.77], #初期速度[x, y].根っこの初期速度ではなく、最初に追跡するとき根っこがどこに移動するか
    "speed_range":[0.8, 1.2], #現在のスピードで移動しうる範囲。下限:現在のスピード*0.8、上限:現在のスピード*1.2など
    "angle":10, #成長する上限角度、弧度法。10度のとき左右5度の範囲に絞る
    "appearance_frame": 5., #根っこが初めて出現するフレーム
    "speed_lower": 0.3, #1フレーム間での成長速度の下限。ノイズか根っこかの判断のために使用
}


model = UNet_3D(in_channels=3,
                noise_strength=param_dict["detection_th"],
                n_classes=1,
                delay_upsample=0).cuda(device)

model_path = "models/model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
print("モデルロード成功")

torch.backends.cudnn.benchmark = True
# 動画取得
video_length, last_image, test_loader = data_info(video_path)

# init setting
tracks = []

tracker = Coord2TrackKalman(move_speed=[param_dict["speed_upper1"], param_dict["speed_upper2"]], init_move=param_dict["init_speed"])

from torch.cuda.amp import autocast

# tracker.update([torch.empty([0, 3])])
for batch_idx, inputs in tqdm(enumerate(test_loader), total=video_length//16, leave=False):
    # inputs.shape = [1, 3, H, W]
    inputs = inputs.cuda(device, non_blocking=True)
    with autocast():
        _, coord = model.get_coord(inputs)
    tracker.update(coord)

tracks = tracker(video_length, delay=1) #.cpu().numpy()
tracks = global_association_solver(tracks, param_dict)

np.savetxt(f"track.txt", tracks, delimiter=',', fmt=["%d", "%d", "%d", "%d", "%.2f"])

tracks = TrackInterpolate(tracks)

track_img = VisualizationTrajectory(last_image, tracks)
track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB)
track_img = Image.fromarray(track_img)
display(track_img)

