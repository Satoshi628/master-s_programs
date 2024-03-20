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
    parser = argparse.ArgumentParser(description="追跡結果の可視化")

    # 引数を追加
    parser.add_argument("-s", "--seg_path", type=str, help="セグメンテーションパス")
    parser.add_argument("-t", "--track_path", type=str, help="追跡結果パス")

    # 引数を解析
    args = parser.parse_args()
    return args

############## dataloader function ##############
def dataload(args):
    #画像はゼロ埋めされていないと順番通りにならないです
    images_path = sorted(glob.glob(os.path.join(args.seg_path, "*.*")))
    return images_path


############## plot function ##############
def plot(images_path, tracks, args):
    
    H,W = cv2.imread(images_path[0]).shape[:2]

    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('track.mp4', fmt, 10., (H,W))

    track_id_unique = np.unique(tracks[:,1])
    colors = {str(tid):np.random.randint(0,200,[3]).tolist() for tid in track_id_unique}

    for frame, path in tqdm(enumerate(images_path), total=len(images_path), leave=False):
        #結局連結成分ラベリングやるから色付きじゃなくて良かった。
        #細胞部分が255、背景が0なら大丈夫です
        image = cv2.imread(path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #jpgだとマスク周りに0以外の部分ができるため下のコードが使えない！！！！！
        # image = ((image != 0)*255).astype(np.uint8)
        #jpg用
        # image = ((image > 20)*255).astype(np.uint8)
        #png用
        image = ((image != 0)*255).astype(np.uint8)
        # _, _, stats, centroids = cv2.connectedComponentsWithStats(gray_img)
        gray_img = np.tile(gray_img[:,:,None], (1,1,3))
        #現在の追跡を取得
        one_img_track = tracks[tracks[:, 0] == frame]
        ### Plot ###
        for coord in one_img_track:
            cv2.putText(gray_img,
                        text='{}'.format(coord[1]),
                        org=(coord[2], coord[3]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3,
                        color=colors[str(coord[1])],
                        thickness=1,
                        lineType=cv2.LINE_4)
        writer.write(gray_img)
    writer.release()

def main():
    args = get_argparse()
    # data load
    images_path = dataload(args)
    tracks = np.loadtxt(args.track_path, delimiter=',').astype(np.int32)


    plot(images_path, tracks, args)



############## main ##############
if __name__ == '__main__':
    main()
