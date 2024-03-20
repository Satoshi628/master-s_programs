#coding: utf-8
#----- Standard Library -----#
import os
import argparse

#----- Public Package -----#
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm

#----- Module -----#
from tracker.tracker import Tracker, global_association_solver, TrackInterpolate

def get_argparse():
    # 引数解析器を作成
    parser = argparse.ArgumentParser(description="ゼニゴケ追跡プロブラム。カメラ固定、分子移動しない前提")

    # 引数を追加
    parser.add_argument("-p", "--seg_path", type=str, help="セグメンテーションパス")
    parser.add_argument("-s", "--speed", type=float, nargs='+', help="1,2frameで移動する距離(px)", default=[10., 15.])

    # 引数を解析
    args = parser.parse_args()
    return args

############## dataloader function ##############
def dataload(args):
    #画像はゼロ埋めされていないと順番通りにならないです
    images_path = sorted(glob.glob(os.path.join(args.seg_path, "*.*")))
    return images_path


############## test function ##############
def test(images_path, args):
    #1,2frameで移動しうるpx数を指定。小さいと追跡が厳しくなる(途切れやすくなる)
    tracker = Tracker(move_speed=args.speed)

    #[H,W, 3]
    sample_img = cv2.imread(images_path[0])

    x = np.linspace(0, sample_img.shape[1], sample_img.shape[1])
    y = np.linspace(0, sample_img.shape[1], sample_img.shape[0])
    x, y = np.meshgrid(x, y)
    mesh = np.stack([x,y],axis=0)

    for frame, path in tqdm(enumerate(images_path), total=len(images_path), leave=False):
        #結局連結成分ラベリングやるから色付きじゃなくて良かった。
        #細胞部分が255、背景が0なら大丈夫です
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #jpgだとマスク周りに0以外の部分ができるため下のコードが使えない！！！！！
        # image = ((image != 0)*255).astype(np.uint8)
        #jpg用
        # image = ((image > 20)*255).astype(np.uint8)
        #png用
        image = ((image != 0)*255).astype(np.uint8)

        _, _, stats, centroids = cv2.connectedComponentsWithStats(image)
        centroids = torch.from_numpy(centroids)

        #tracker.updateの引数は検出結果です。
        #時系列はリスト、1つ1つの検出結果はtorch.tensor([N,2(x,y)])で入力してください。
        #例
        #3つの時系列データを入力する際
        #[torch.empty([12,2]), torch.empty([10,2]), torch.empty([14,2])]
        tracker.update([centroids])

    #2段階追跡結果
    #ちなみに2フレーム以下しか続いていない追跡は削除しています
    #length_thで設定できます
    tracks = tracker(len(images_path), length_th=2)
    print(tracks.shape)
    #大域的追跡結果、使わなくていいかも
    # tracks = global_association_solver(tracks)
    tracks = tracks.numpy()
    #補正された追跡を生成
    TrackInterpolate(tracks)
    #追跡結果は[N,4(frame,id,x,y)]で保存されます

    np.savetxt(f"track.txt", tracks, delimiter=',', fmt=["%d", "%d", "%d", "%d"])


def main():
    args = get_argparse()
    # data load
    images_path = dataload(args)


    test(images_path, args)



############## main ##############
if __name__ == '__main__':
    main()
