
#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import sys
import random
import glob
#----- 専用ライブラリ -----#
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import cv2
#----- 自作モジュール -----#
#None

"""
hdf5のフォルダ構造は
hdf5┬exp0F0001┬0001┬0000
    │         │    ├0001
    │         │    …
    │         │    └1011
    │         ├0003
    │         …
    …
    └exp0F00016
"""

"""
C2C12DataLoaderの仕様書
train、evalにおいて使うデータを指定できるようにする。
複数のデータを使えるようにつくる。
複数のMPM(時差1,3,5など)を使えるようにする。
時差はrandamで指定する。
transformを使用できるようにする。
多分Segmentationと同じなのでCovidを流用できるはず
"""


##### C2C12 dataset #####
class C2C12Loader(data.Dataset):
    # 初期設定
    def __init__(self, use_data_path, Annotater, use_iter=[1], transform=None):
        """C2C12データセット、MPMLoader

        Args:
            use_data_path (list): 使用するデータのパス。複数可
            Annotater (str): 使用するMPMのAnnotater
            use_iter (list, optional): 使用するiteretationフレーム差を示す。複数の場合ランダムに選ぶ. Defaults to [1].
            transform (transform), optional): data Augmentation. Defaults to None.
        """        
        use_paths = sorted(use_data_path)
        img_paths = []
        mpm_paths = []
        for path in use_paths:
            #無理やり相対パスも挿入する
            img_paths.append(list(map(lambda x: os.path.join(path, x), sorted(os.listdir(path)))))
            #imgパスからmpmパスを取得する
            mpm_paths.append(re.search(r'exp\d{1}_F\d{4}', path).group())
        
        self.mpmf = h5py.File("/mnt/kamiya/dataset/C2C12/data{}/Cell_{}_MPM.hdf5".format(mpm_paths[0][3], Annotater), mode='r')
        #mpm_dirsは3次元配列
        #1次元はexp0_Fnumberを示す。2次元はiterを示す。3次元は画像数を示す。
        mpm_dirs = []
        for path in mpm_paths:
            group = self.mpmf[path]
            #use_iterで指定したファイルのみのパスを抽出
            mpm_dirs.append([[os.path.join(value.name, key) for key in value.keys()] for value in group.values() if int(value.name[-4:]) in use_iter])
        
        #__getitem__で使う変数を定義する。
        #２次配列を展開
        self.img_list = [item2 for item1 in img_paths for item2 in item1]

        #mpm_listは、一次元目はデータ数が入り、二次元目にはそのデータで可能なitv(mpmマップ)が入っている
        self.mpm_list = []
        for img_path, mpm_iter in zip(img_paths, mpm_dirs):
            #img_pathの数分の2次空配列を作製
            mpm_path_tmp = [[] for _ in img_path]
            for mpm_path in mpm_iter:
                for idx, mpm in enumerate(mpm_path):
                    mpm_path_tmp[idx].append(mpm)
            self.mpm_list.extend(mpm_path_tmp)
        
        #getitemのindexとdata_dictのindexを合わせる配列
        self.idx_transfer = [i for i in range(len(self.img_list))]
        none_idx = [idx for idx, item in enumerate(self.mpm_list) if not item]
        self.idx_transfer = [item for item in self.idx_transfer if not item in none_idx]
        #iter保存
        self.iter = use_iter

        #変換処理保存
        self.transform = transform
        #print(len(self.img_list))
        #print(len(self.mpm_list))
        #print(len(self.idx_transfer))
        #for item in self.mpm_list:
        #    print(len(item)) if len(item) !=5 else ...
        #input()


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        mpm_number = random.randrange(len(self.mpm_list[data_index]))
        itv = self.iter[mpm_number]

        image1_name = self.img_list[data_index]
        image2_name = self.img_list[data_index + itv]
        mpm_name = self.mpm_list[data_index][mpm_number]

        # 画像読み込み  # Image.open = 読み込み .convert("RGB") = RGBで読み込み
        image1 = Image.open(image1_name).convert("I")
        image2 = Image.open(image2_name).convert("I")
        label = self.mpmf[mpm_name][...]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image1, image2, label)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)


##### OIST dataset #####
class OISTLoader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning="GFP", mode="train", split=0, use_iter=[1], transform=None):
        """OISTデータセット、MPMLoader

        Args:
            use_data_path (list): 使用するデータのパス。複数可
            Annotater (str): 使用するMPMのAnnotater
            use_iter (list, optional): 使用するiteretationフレーム差を示す。複数の場合ランダムに選ぶ. Defaults to [1].
            transform (transform), optional): data Augmentation. Defaults to None.
        """        
        use_data_list = {
                        "GFP":
                            ["Sample1/video",
                            "Sample2/video",
                            "Sample3/video",
                            "Sample4/video",
                            "Sample5/video"],
                        "GFP_Low":
                            ["Sample6/video",
                            "Sample7/video",
                            "Sample8/video",
                            "Sample9/video",
                            "Sample10/video"],
                        "TMR":
                            ["Sample11/video",
                            "Sample12/video",
                            "Sample13/video",
                            "Sample14/video",
                            "Sample15/video"],
                        "SF650":
                            ["Sample16/video",
                            "Sample17/video",
                            "Sample18/video",
                            "Sample19/video",
                            "Sample20/video"]}

        use_frame_list = {
                    "GFP": [[0, 100], [0, 100], [0, 100], [0, 100], [0, 100]],
                    "GFP_Low": [[0, 100], [0, 100], [0, 100], [0, 100], [0, 100]],
                    "TMR": [[0, 100], [0, 100], [0, 100], [0, 100], [0, 100]],
                    "SF650": [[0, 100], [0, 100], [0, 100], [0, 100], [0, 100]]}
        
        split_list = [{"train": [0, 1, 2], "val": [3], "test": [4]},
                 {"train": [4, 0, 1], "val": [2], "test": [3]},
                 {"train": [3, 4, 0], "val": [1], "test": [2]},
                 {"train": [2, 3, 4], "val": [0], "test": [1]},
                 {"train": [1, 2, 3], "val": [4], "test": [0]}]

        #エラー対策
        if not Staning in use_data_list:  #キーがあるかどうか
            raise KeyError("Staningが{0},{1},{2},{3}以外の文字になっています。".format(*list(use_data_list.keys())))
        if not mode in split_list[0]:  # キーがあるかどうか
            raise KeyError("modeが{0},{1},{2}以外の文字になっています。".format(*list(split_list[0].keys())))

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, use_data_list[Staning][idx]) for idx in split_list[split][mode]]
        use_frames = [use_frame_list[Staning][idx] for idx in split_list[split][mode]]

        img_paths = []
        MPM_paths = []

        for path, frame in zip(use_paths, use_frames):
            #相対パスを追加
            img_paths.append(sorted(glob.glob(path + "/*.png"))[frame[0] : frame[1]])
            #imgパスからmpmパスを取得する
            MPM_paths.append(re.search(r'Sample\d+', path).group())

        #読み込み時すべてnumpyより、list内部numpyの方が処理が早い
        #image保存、正規化も行う(0~1の値にする)、大きさ[1,H,W]となる。
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        #Cell Point保存
        CP_data = [h5py.File("{}/Cell_Point_Annotation.hdf5".format(root_dir), mode='r')[path][...] for path in MPM_paths]
        #特定フレームの抽出のためのflag
        frame_flag = [(frame[0] <= data[:, 0]) & (data[:, 0] < frame[1]) for data, frame in zip(CP_data, use_frames)]
        CP_data = [data[flag] for data, flag in zip(CP_data, frame_flag)]
        
        #frameのかぶりがある可能性があるため結合前にcountする
        count = np.concatenate([np.unique(data[:, 0], return_counts=True)[-1] for data in CP_data])
        #結合
        self.CP_data = np.concatenate(CP_data)
        self.CP_data = torch.tensor(self.CP_data)

        #MPM保存
        MPM_images = [h5py.File("{}/Molecule_MPM.hdf5".format(root_dir), mode='r')[path] for path in MPM_paths]
        #MPM_imagesは、一次元目はデータ数が入り、二次元目にはそのデータで可能なitv(mpmマップ)が入っている
        MPM_images = [[[item[...]
                        for item in MPM[str(itv).zfill(4)].values()]  #↑
                        for itv in use_iter]    #↑
                        for MPM in MPM_images]  #↑
        self.MPM_images = []
        for idx, path in enumerate(MPM_images):
            #img_pathの数分の2次空配列を作製
            mpm_path_tmp = [[] for _ in range(use_frames[idx][1] - use_frames[idx][0])]
            for mpm_path in path:
                for path_idx, mpm in enumerate(mpm_path):
                    mpm_path_tmp[path_idx].append(mpm)
            self.MPM_images.extend(mpm_path_tmp)
        
        #getitemのindexとdata_dictのindexを合わせる配列
        self.idx_transfer = [i for i in range(len(self.images))]
        none_idx = [idx for idx, item in enumerate(self.MPM_images) if not item]
        self.idx_transfer = [item for item in self.idx_transfer if not item in none_idx]
        
        #iter保存
        self.iter = use_iter

        
        #変換処理保存
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        mpm_number = random.randrange(len(self.MPM_images[data_index]))
        itv = self.iter[mpm_number]

        # 画像読み込み
        image1 = self.images[data_index]
        image2 = self.images[data_index + itv]
        #label.size() => [3(x,y,t),H,W]
        label = self.MPM_images[data_index][mpm_number]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image1, image2, label)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)


##### OIST dataset #####
class OISTLoader_low_test(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov", Staning="GFP", split=0, use_iter=[1], transform=None):
        """OISTデータセット、MPMLoader

        Args:
            use_data_path (list): 使用するデータのパス。複数可
            Annotater (str): 使用するMPMのAnnotater
            use_iter (list, optional): 使用するiteretationフレーム差を示す。複数の場合ランダムに選ぶ. Defaults to [1].
            transform (transform), optional): data Augmentation. Defaults to None.
        """        
        use_data_list = {
                        "GFP":
                            ["Sample1/video",
                            "Sample2/video",
                            "Sample3/video"],
                        "GFP_Low":
                            ["Sample4/video",
                            "Sample5/video",
                            "Sample6/video"],
                        "TMR":
                            ["Sample7/video",
                            "Sample8/video",
                            "Sample9/video"],
                        "SF650":
                            ["Sample10/video",
                            "Sample11/video",
                            "Sample12/video"]}

        use_frame_list = {
                    "GFP": [[0, 100], [0, 100], [0, 100]],
                    "GFP_Low": [[0, 100], [0, 100], [0, 100]],
                    "TMR": [[0, 100], [0, 100], [0, 100]],
                    "SF650": [[0, 100], [0, 100], [0, 100]]}
        
        split_list = [{"val": [2], "test": [0]},
                 {"val": [0], "test": [1]},
                 {"val": [1], "test": [2]}]

        #エラー対策
        if not Staning in use_data_list:  #キーがあるかどうか
            raise KeyError("Staningが{0},{1},{2},{3}以外の文字になっています。".format(*list(use_data_list.keys())))

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, use_data_list[Staning][idx]) for idx in split_list[split]["test"]]
        use_frames = [use_frame_list[Staning][idx] for idx in split_list[split]["test"]]

        img_paths = []
        MPM_paths = []

        for path, frame in zip(use_paths, use_frames):
            #相対パスを追加
            img_paths.append(sorted(glob.glob(path + "/*.png"))[frame[0] : frame[1]])
            #imgパスからmpmパスを取得する
            MPM_paths.append(re.search(r'Sample\d+', path).group())

        #読み込み時すべてnumpyより、list内部numpyの方が処理が早い
        #image保存、正規化も行う(0~1の値にする)、大きさ[1,H,W]となる。
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        #Cell Point保存
        CP_data = [h5py.File("{}/Cell_Point_Annotation.hdf5".format(root_dir), mode='r')[path][...] for path in MPM_paths]
        #特定フレームの抽出のためのflag
        frame_flag = [(frame[0] <= data[:, 0]) & (data[:, 0] < frame[1]) for data, frame in zip(CP_data, use_frames)]
        CP_data = [data[flag] for data, flag in zip(CP_data, frame_flag)]
        
        #frameのかぶりがある可能性があるため結合前にcountする
        count = np.concatenate([np.unique(data[:, 0], return_counts=True)[-1] for data in CP_data])
        #結合
        self.CP_data = np.concatenate(CP_data)
        self.CP_data = torch.tensor(self.CP_data)

        #MPM保存
        MPM_images = [h5py.File("{}/Molecule_MPM.hdf5".format(root_dir), mode='r')[path] for path in MPM_paths]
        #MPM_imagesは、一次元目はデータ数が入り、二次元目にはそのデータで可能なitv(mpmマップ)が入っている
        MPM_images = [[[item[...]
                        for item in MPM[str(itv).zfill(4)].values()]  #↑
                        for itv in use_iter]    #↑
                        for MPM in MPM_images]  #↑
        self.MPM_images = []
        for idx, path in enumerate(MPM_images):
            #img_pathの数分の2次空配列を作製
            mpm_path_tmp = [[] for _ in range(use_frames[idx][1] - use_frames[idx][0])]
            for mpm_path in path:
                for path_idx, mpm in enumerate(mpm_path):
                    mpm_path_tmp[path_idx].append(mpm)
            self.MPM_images.extend(mpm_path_tmp)
        
        #getitemのindexとdata_dictのindexを合わせる配列
        self.idx_transfer = [i for i in range(len(self.images))]
        none_idx = [idx for idx, item in enumerate(self.MPM_images) if not item]
        self.idx_transfer = [item for item in self.idx_transfer if not item in none_idx]
        
        #iter保存
        self.iter = use_iter

        
        #変換処理保存
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        mpm_number = random.randrange(len(self.MPM_images[data_index]))
        itv = self.iter[mpm_number]

        # 画像読み込み
        image1 = self.images[data_index]
        image2 = self.images[data_index + itv]
        #label.size() => [3(x,y,t),H,W]
        label = self.MPM_images[data_index][mpm_number]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image1, image2, label)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)


##### PTC dataset #####
class PTCLoader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/PTC", Staning="RECEPTOR",Density="Low", mode="train", split=0, use_iter=[1], transform=None):    
        use_data_list = {
                        "Low":
                            ["Sample1",
                            "Sample2",
                            "Sample3",
                            "Sample4",
                            "Sample5"],
                        "Mid":
                            ["Sample6",
                            "Sample7",
                            "Sample8",
                            "Sample9",
                            "Sample10"],
                        "High":
                            ["Sample11",
                            "Sample12",
                            "Sample13",
                            "Sample14",
                            "Sample15"],
                        "Mid2":
                            ["Sample16",
                            "Sample17",
                            "Sample18",
                            "Sample19",
                            "Sample20"]}
        
        split_list = [{"train": [0, 1, 2], "val": [3], "test": [4]},
                 {"train": [4, 0, 1], "val": [2], "test": [3]},
                 {"train": [3, 4, 0], "val": [1], "test": [2]},
                 {"train": [2, 3, 4], "val": [0], "test": [1]},
                 {"train": [1, 2, 3], "val": [4], "test": [0]}]

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, Staning, use_data_list[Density][idx]) for idx in split_list[split][mode]]

        img_paths = []
        MPM_paths = []

        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(path + "/*.tif")))
            #imgパスからmpmパスを取得する
            MPM_paths.append(re.search(r'Sample\d+', path).group())

        #読み込み時すべてnumpyより、list内部numpyの方が処理が早い
        #image保存、正規化も行う(0~1の値にする)、大きさ[1,H,W]となる。
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        #Cell Point保存
        CP_data = [h5py.File("{}/{}/Cell_Point_Annotation.hdf5".format(root_dir, Staning), mode='r')[path][...] for path in MPM_paths]
        
        #frameのかぶりがある可能性があるため結合前にcountする
        count = np.concatenate([np.unique(data[:, 0], return_counts=True)[-1] for data in CP_data])
        #結合
        self.CP_data = np.concatenate(CP_data)
        self.CP_data = torch.tensor(self.CP_data)

        #MPM保存
        MPM_images = [h5py.File("{}/{}/Molecule_MPM.hdf5".format(root_dir, Staning), mode='r')[path] for path in MPM_paths]
        #MPM_imagesは、一次元目はデータ数が入り、二次元目にはそのデータで可能なitv(mpmマップ)が入っている
        MPM_images = [[[item[...]
                        for item in MPM[str(itv).zfill(4)].values()]  #↑
                        for itv in use_iter]    #↑
                        for MPM in MPM_images]  #↑
        self.MPM_images = []
        for idx, path in enumerate(MPM_images):
            #img_pathの数分の2次空配列を作製
            mpm_path_tmp = [[] for _ in range(100)]
            for mpm_path in path:
                for path_idx, mpm in enumerate(mpm_path):
                    mpm_path_tmp[path_idx].append(mpm)
            self.MPM_images.extend(mpm_path_tmp)
        
        #getitemのindexとdata_dictのindexを合わせる配列
        self.idx_transfer = [i for i in range(len(self.images))]
        none_idx = [idx for idx, item in enumerate(self.MPM_images) if not item]
        self.idx_transfer = [item for item in self.idx_transfer if not item in none_idx]
        #iter保存
        self.iter = use_iter

        
        #変換処理保存
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        mpm_number = random.randrange(len(self.MPM_images[data_index]))
        itv = self.iter[mpm_number]

        # 画像読み込み
        image1 = self.images[data_index]
        image2 = self.images[data_index + itv]
        #label.size() => [3(x,y,t),H,W]
        label = self.MPM_images[data_index][mpm_number]

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image1, image2, label)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)



def OISTLoader_raw_data(path="/mnt/kamiya/code/my_exp/My_MPM2/Mid_GFP_v9.avi", frame=None):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        # 正常に読み込めたのかチェックする
        # 読み込めたらTrue、失敗ならFalse
        print("動画の読み込み失敗")
        sys.exit()

    images = []
    
    while True:
        # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
        is_image, frame_img = cap.read()
        if is_image:
            # 画像を保存
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            images.append(torch.from_numpy(frame_img))
        else:
            # フレーム画像が読込なかったら終了
            break
        if frame is not None:
            frame -= 1
            if frame < 0:
                break

        if len(images) == 2:
            img = torch.stack(images, dim=0) / 255.
            images = [images[-1]]
            yield img[None]
    
    append_num = 2 - len(images)
    images = images + [images[-1] for _ in range(append_num)]

    img = torch.stack(images, dim=0) / 255.
    yield img[None]

