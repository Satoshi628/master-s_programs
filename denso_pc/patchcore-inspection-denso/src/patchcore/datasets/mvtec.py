import os
from enum import Enum

import glob
import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
_SCENARIO = {
    "bottle":["broken_large","broken_small","contamination"],
    "cable":["bent_wire","cable_swap","combined","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"],
    "capsule":["crack","faulty_imprint","poke","scratch","squeeze"],
    "carpet":["color","cut","hole","metal_contamination","thread"],
    "grid":["bent","broken","glue","metal_contamination","thread"],
    "hazelnut":["crack","cut","hole","print"],
    "leather":["color","cut","fold","glue","poke"],
    "metal_nut":["bent","color","flip","scratch"],
    "pill":["color","combined","contamination","crack","faulty_imprint","pill_type","scratch"],
    "screw":["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"],
    "tile":["crack","glue_strip","gray_stroke","oil","rough"],
    "toothbrush":["defective"],
    "transistor":["bent_lead","cut_lead","damaged_case","misplaced"],
    "wood":["color","combined","hole","liquid","scratch"],
    "zipper":["broken_teeth","combined","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"],
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


class MVTecDataset_Scenario(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            classname,
            scenario,
            broken_type,
            resize=256,
            imagesize=224,
            split=DatasetSplit.TRAIN,
            train_val_split=1.0,
            **kwargs,
        ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            scenario: [str]. Normal or N2A or A2N.
            scenario: [str]. broken type.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = classname
        self.scenario = scenario # Normal or N2A or A2N
        if self.scenario == "A2N":
            assert broken_type in _SCENARIO[classname] is not None, "broken_type Error"
            self.broken_type = broken_type
        else:
            self.broken_type = None
        self.train_val_split = train_val_split

        self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if mask_path is not None and anomaly != self.broken_type:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly in ["good", self.broken_type]),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        image_paths = os.path.join(self.source, self.classnames_to_use, self.split.value, "*", "*.png")
        anomaly_types = os.path.join(self.source, self.classnames_to_use, "ground_truth")
        
        image_paths = sorted(glob.glob(image_paths))
        anomaly_types = sorted(os.listdir(anomaly_types))

        data_to_iterate = []
        for path in image_paths:
            if path.split("/")[-2] == "good":
                mask_path = None
            else:
                mask_path = path.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")

            anomaly = path.split("/")[-2]
            data_to_iterate.append([self.classnames_to_use, anomaly, path, mask_path])

        if self.scenario == "A2N":
            A2N_image_paths = os.path.join(self.source, self.classnames_to_use, "test", self.broken_type, "*.png")
            A2N_image_paths = sorted(glob.glob(A2N_image_paths))
            A2N_image_paths = A2N_image_paths[:len(A2N_image_paths)//2]
            if self.split == DatasetSplit.TRAIN:
                for path in A2N_image_paths:
                    mask_path = path.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")
                    anomaly = path.split("/")[-2]
                    data_to_iterate.append([self.classnames_to_use, anomaly, path, mask_path])
            
            else:
                data_to_iterate = [[classname, anomaly, image_path, mask_path] for classname, anomaly, image_path, mask_path in data_to_iterate if not image_path in A2N_image_paths]

        if self.scenario == "N2A":
            if self.split == DatasetSplit.TRAIN:
                data_to_iterate = [[classname, anomaly, image_path, mask_path] for classname, anomaly, image_path, mask_path in data_to_iterate if not "pseudo_" in image_path]
        if self.scenario == "Normal" and self.split == DatasetSplit.TEST:
            data_to_iterate_temp = []
            for classname, anomaly, image_path, mask_path in data_to_iterate:
                if anomaly == "pseudo_anomaly":
                    mask_path = None
                data_to_iterate_temp.append([classname, anomaly, image_path, mask_path])
            data_to_iterate = data_to_iterate_temp
        return data_to_iterate


if __name__ == "__main__":
    dataset = MVTecDataset_Scenario(
        "/mnt/kamiya/dataset/MVtec_AD-N2A",
        "bottle",
        "Normal",
        None,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TEST,
        train_val_split=1.0,)
    for data in dataset:
        print(data["mask"].sum())
        print(data["image_path"])
        input()