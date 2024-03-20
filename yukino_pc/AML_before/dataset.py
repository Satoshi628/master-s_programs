import os
import torch
import numpy as np
import random
import glob
import torch
from numpy import load
from pathlib import Path
from PIL import Image


############################# データローダー ########################################


class FlyCellDataLoader_crossval():
    def __init__(self, rootdir="", val_area="2", split='train', iteration_number=None):
        self.split = split
        self.training = True if split == 'train' else False

        filelist_train = []
        filelist_val = []
        filelist_test = []
        test_area = val_area + 1 if val_area != 5 else 1

        for i in range(1, 6):
            dataset = [os.path.join(f'{rootdir}/5-fold/Area_{i}/{data}')
                       for data in os.listdir(f'{rootdir}/5-fold/Area_{i}')]
            if i == val_area:
                filelist_test = filelist_test + dataset
            elif i == test_area:
                filelist_val = filelist_val + dataset
            else:
                filelist_train = filelist_train + dataset
        

        if split == 'train':
            self.filelist = filelist_train
        elif split == 'val':
            self.filelist = filelist_val
        elif split == 'test':
            self.filelist = filelist_test

        if self.training:
            self.number_of_run = 1
            self.iterations = iteration_number
        else:
            self.number_of_run = 16
            self.iterations = None

        print(f'val_area : {val_area} test_area : {test_area} ', end='')
        print(f"{split} files : {len(self.filelist)}")

    def __getitem__(self, index):
        # print(index)
        if self.training:
            index = random.randint(0, len(self.filelist) - 1)
            dataset = self.filelist[index]
            #print('train',dataset)
            #print('train', len(dataset))  # 63

        else:
            dataset = self.filelist[index // self.number_of_run]
            #print('val', dataset)
            #print('val', len(dataset))  

        # load files
        filename_data = os.path.join(dataset)
        inputs = np.load(filename_data)
        #4(12)枚の大きな画像から256*256の画像を生成
        #4(12)*(1024/256)*(1024/256)=64(192)
        #実質64(192)枚学習
        # print(len(inputs))   #1024
        # print(inputs.shape)  # (1024, 1024, 2)
        # print(inputs.dtype)  # float32

        # split features labels
        if self.training:
            x = random.randint(0, inputs.shape[0] - 256)
            y = random.randint(0, inputs.shape[0] - 256)
        else:
            x = index % self.number_of_run//4 * 256
            y = index % self.number_of_run % 4 * 256

        features = inputs[x:x+256, y:y+256,
                          0:1].transpose(2, 0, 1).astype(np.float32)
        features /= 255.0
        
        labels = inputs[x: x + 256, y: y + 256, -1].astype(int)
        
        
        #print(features.shape)  # (1, 256, 256)
        #print(labels.shape)  # (256, 256)

        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(labels).long()
        #print("{:.2f}".format(torch.max(fts))) #<=1
        #print("{:.2f}".format(torch.min(fts)))  # 0

        #print("{:.2f}".format(torch.max(lbs)))  # 4
        #print("{:.2f}".format(torch.min(lbs)))  # 0

        #print(fts.shape)

        return fts, lbs

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations


################################ データ内容  #######################################
"""・データセットは1024×1024の細胞画像が20枚です(gray scaleの1ch)
・5回の交差検証用データセットです
・1ブロック4枚の合計5ブロックに分けています。（Area1 ～ Area5, 詳しくはdata.png参照)
    ・学習データ：4枚×3ブロック. 1枚から256×256にdataloader内でランダムクロップ
    ・検証データ：4枚×1ブロック. 1枚を16分割(256×256)で固定
    ・テストデータ：4枚×1ブロック. 1枚を16分割(256×256)で固定
    ・image shape: (bs, 256, 256, 1) label shape: (bs, 256, 256)

    ・クラス数: 5クラス
    class 0: membrane
    class 1: mitochondria
    class 2: synapse
    class 3: glia/extracellular
    class 4: intracellular


    詳しくはここ↓
    https: // github.com/unidesigner/groundtruth-drosophila-vnc

    ※注意※
    マスクから正解ラベルを作ると
    ・最大4クラスになる
    ・マスク値が0or255になっていない
    という問題があります。"""

################################ 使い方 #######################################


"""
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--rootdir", type=str, default='data')
        parser.add_argument("--batchsize", "-b", default=16, type=int)
        # iter: イテレーション数(1epoch内で何回ランダムクロップを繰り返すか)
        parser.add_argument("--iter", default=12, type=int)
        # threads:同時に使っているGPUの数
        parser.add_argument("--threads", default=2, type=int)
        parser.add_argument("--val_area", default=1, type=int,
                            help='cross-val test area [default: 5]')
        args = parser.parse_args()

        ds_train = FlyCellDataLoader_crossval(
            rootdir=args.rootdir, val_area=args.val_area, split='train', iteration_number=args.batchsize*args.iter)
        ds_val = FlyCellDataLoader_crossval(
            rootdir=args.rootdir, val_area=args.val_area, split='val')
        ds_test = FlyCellDataLoader_crossval(
            rootdir=args.rootdir, val_area=args.val_area, split='test')

        train_loader = torch.utils.data.DataLoader(
            ds_train, batch_size=args.batchsize, shuffle=True, num_workers=args.threads)
        val_loader = torch.utils.data.DataLoader(
            ds_val, batch_size=args.batchsize, shuffle=False, num_workers=args.threads)
        test_loader = torch.utils.data.DataLoader(
            ds_test, batch_size=args.batchsize, shuffle=False, num_workers=args.threads)"""

############## 説明 ########################
"""rootdir: データセットのディレクトリ名
    batchsize: バッチサイズ
    iter: イテレーション数(1epoch内で何回ランダムクロップを繰り返すか)
    threads: num_workers数
    val_area: validationのブロック番号(１～５を選択). これを変えることで5回のCross Validationが可能


    これらの値をdefaultで固定, かつ乱数を以下↓に設定して固定することで研究室内ランキング戦に参加できます。
    皆様のご参加お待ちしています！！！！！

    ### この3行, プログラムのdataloaderの前に追加してネ ###
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
"""
