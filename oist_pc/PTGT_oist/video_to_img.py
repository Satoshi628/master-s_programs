#coding: utf-8
#----- Standard Library -----#
import sys
import os

#----- Public Package -----#
import glob
import cv2

#----- Module -----#

video_paths = glob.glob("./data/**/*.avi")

for path in video_paths:
    save_dir = os.path.join(*path.split("/")[:-1], "video")
    print(save_dir)
    
    # 動画ファイル読込
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        # 正常に読み込めたのかチェックする
        # 読み込めたらTrue、失敗ならFalse
        print("動画の読み込み失敗")
        sys.exit()
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # print("width:{}, height:{}, count:{}, fps:{}".format(width,height,count,fps))

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    while True:
        # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
        is_image, frame_img = cap.read()
        if is_image:
            # 画像を保存
            cv2.imwrite(save_dir + "/" + str(n).zfill(digit) + ".png", frame_img)
        else:
            # フレーム画像が読込なかったら終了
            break
        n += 1

    cap.release()
