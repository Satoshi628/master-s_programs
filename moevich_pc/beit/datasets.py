# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import numpy as np
import torch
import glob
from PIL import Image


from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform

from dall_e.utils import map_pixels
from masking_generator import MaskingGenerator
from dataset_folder import ImageFolder


class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN,
                    std=IMAGENET_INCEPTION_STD,
                ),
            ])
        else:
            raise NotImplementedError()

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_beit_pretraining_dataset(args):
    transform = DataAugmentationForBEiT(args)
    print("Data Aug = %s" % str(transform))
    if args.use_NII:
        check_path = check_split_image(args.split_paths, args.split)
        dataset = ImageFolder(args.data_path, transform=transform, is_valid_file=check_path)
    elif args.use_Eye:
        dataset = ImageFolder(args.data_path, transform=transform)
    else:
        dataset = ImageFolder(args.data_path, transform=transform)
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "NII":
        split = "train" if is_train else "val"
        root = args.data_path
        dataset = Data_Loader(root, classes=args.nb_classes, dataset_type=split, transform=transform)
        nb_classes = args.nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class check_split_image():
    def __init__(self, split_dirs, split):

        ture_folders = []
        for split_dir in split_dirs:
            #使用するデータのパスを取得
            split_data = np.loadtxt(split_dir, dtype=str, delimiter=",")
            #path情報抽出
            data_paths = split_data[split_data[:, 1] == split, 0]
            ture_folders.extend([path.split("/")[-1] for path in data_paths])

        self.ture_folders = ture_folders
    def __call__(self, path):
        if "images" not in path:
            return False

        return path.split("/")[-3] in self.ture_folders



##### 2class NDPI dataset #####
class Data_Loader(torch.utils.data.Dataset):
    # 初期設定
    def __init__(self, split_dir, classes=2, dataset_type='train', transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #使用するデータのパスを取得
        split_data = np.loadtxt(split_dir, dtype=str, delimiter=",")
        #path情報抽出
        data_paths = split_data[split_data[:, 1] == dataset_type, 0]

        image_paths = []
        label_paths = []
        label_data = []
        for path in data_paths:
            if len(glob.glob(os.path.join(path, "images", "*.png"))) == 0:
                continue
            image_paths.extend(sorted(glob.glob(os.path.join(path, "images", "*.png"))))
            label_paths.extend(sorted(glob.glob(os.path.join(path, "labels", "*.png"))))

            
            annotation_data = np.loadtxt(os.path.join(path, "annotations.txt"), dtype=str, delimiter="\t")[1:, 1:]
            annotation_data = annotation_data.astype(np.float32)
            label_data.append(annotation_data.argmax(axis=-1))
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.label_data = np.concatenate(label_data, axis=0)
        if classes == 2:
            self.label_data[self.label_data == 2] = 1

        # self.dataset_type = dataset = 'train' or 'val' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        label_path = self.label_paths[index]
        # label_img = Image.open(label_path)
        label = self.label_data[index]

        # もしself.transformが有効なら
        if self.transform:
            image = self.transform(image)

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_paths)

    def get_class_count(self):
        label_num, label_count = np.unique(self.label_data, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count

