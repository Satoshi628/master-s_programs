#coding: utf-8
#----- Standard Library -----#
import os
import argparse

#----- Public Package -----#
import numpy as np
from tqdm import tqdm
import glob

#----- Module -----#
#None



def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def mkdat(path, track):
    with open(path, "w") as f:
        for t in track:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")



def main(track_path):
    folder_path = os.path.splitext(track_path)[0]
    mkdir(folder_path)
    
    tracks = np.loadtxt(track_path, delimiter=',').astype(np.int32)
    tracks_uni_ids = np.unique(tracks[:, 1])
    for idx, uni_id in enumerate(tracks_uni_ids):
        one_track = tracks[tracks[:, 1] == uni_id][:, [0, 2, 3]]
        file_path = os.path.join(folder_path, f"{idx+1:05d}.dat")
        mkdat(file_path, one_track)


############## main ##############
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='video_to_img.py',  # プログラム名
        description='description',  # 引数のヘルプの前に表示
        add_help=True,  # -h/–help オプションの追加
    )

    parser.add_argument('-p', '--path', help="data folder")
    # 引数を解析する
    args = parser.parse_args()

    track_paths = sorted(glob.glob(os.path.join(args.path, "*track.txt")))

    for track_path in track_paths:
        print(track_path)
        main(track_path)
