#coding: utf-8
#----- 標準ライブラリ -----#
import sys

#----- 専用ライブラリ -----#
from glob import glob
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
#----- 自作モジュール -----#
# None

# グローバル変数
ANIMAL_LIST = {"American_Pit_Bull_Terrier": 0,
                "Birman": 1,
                "British_Shorthair": 2,
                "Ragdoll": 3,
                "Russian_Blue": 4,
                "Scottish_Terrier": 5,
                "Shiba_Inu": 6,
                "Siamese": 7,
                "Staffordshire_Bull_Terrier": 8,
                "Yorkshire_Terrier": 9}

# グローバル変数
ANIMAL_NAME_JA = ["アメリカン・ピット・ブル・テリア",
                "バーマン",
                "ブリティッシュショートヘア",
                "ラグドール",
                "ロシアンブルー",
                "スコティッシュ・テリア",
                "柴犬",
                "シャム猫",
                "スタッフォードシャー・ブル・テリア",
                "ヨークシャー・テリア"]

# グローバル変数
ANIMAL_NAME_EN = ["American Pit Bull Terrier",
                "Birman",
                "British Shorthair",
                "Ragdoll",
                "Russian Blue",
                "Scottish Terrier",
                "Shiba Inu",
                "Siamese",
                "Staffordshire Bull Terrier",
                "Yorkshire Terrier"]

##### Ghibli dataset #####
class Ghibli_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/content/Dataset_PET/PET", dataset_type='train', transform=None):
        # self.dataset_type = 'train' or 'test'
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス
        self.dataset_type = dataset_type

        image_list = sorted(glob(root_dir + "/*"))

        # train data 10*160  = 1600
        # test data  10*30   = 400

        if self.dataset_type == 'train':
            image_list = [glob(img_paths + "/*")[:170] for img_paths in image_list]
        elif self.dataset_type == 'test':
            image_list = [glob(img_paths + "/*")[170:] for img_paths in image_list]

        # 2次リスト => 一次リスト
        image_list = sorted([img for img_paths in image_list for img in img_paths])

        label_list = [ANIMAL_LIST[img_path.split("/")[-2]] for img_path in image_list]

        self.image_list = [Image.open(img_path).convert("RGB") for img_path in image_list]
        self.label_list = np.array(label_list)

        # self.transform = transform
        self.transform = transform

    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.label_list[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)

    def __len__(self):
        # データの長さ
        return len(self.image_list)


############## dataloader関数 ##############
def dataload(root_dir, train_batch, test_batch):
    if train_batch > 128:
        print("train_batchが大きすぎます")
        print("train_batchは128以下の値にしてください")
        sys.exit(0)
    
    if test_batch > 128:
        print("test_batchが大きすぎます")
        print("test_batchは128以下の値にしてください")
        sys.exit(0)
    

    # data augmentation + preprocceing
    train_transform = transforms.Compose([transforms.RandomCrop(size=(128, 128)),  # ランダムクロップ(128x128)
                                        transforms.RandomHorizontalFlip(),  # ランダムに左右反転
                                        transforms.ToTensor(),  # 0~1正規化+Tensor型に変換
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # データの標準化
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),  # 0~1正規化+Tensor型に変換
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # データの標準化
                                        ])

    # define detaset
    train_dataset = Ghibli_Loader(root_dir=root_dir, dataset_type='train', transform=train_transform)
    test_dataset = Ghibli_Loader(root_dir=root_dir, dataset_type='test', transform=test_transform)

    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを削除

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader
