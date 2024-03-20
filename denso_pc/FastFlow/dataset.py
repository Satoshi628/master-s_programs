import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import constants as const


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

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )

        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                ).convert("L")
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)



class MVTecDataset_Scenario(torch.utils.data.Dataset):
    def __init__(self, root, category, scenario, broken_type, input_size, is_train=True):
        
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.scenario = scenario
        self.category = category
        
        if self.scenario == "A2N":
            assert broken_type in _SCENARIO[category] is not None, "broken_type Error"
            self.broken_type = broken_type
        elif self.scenario == "N2A":
            self.broken_type = "pseudo_anomaly"
        else:
            self.broken_type = None

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        #正常に変更された欠損タイプは"good-"を付ける
        if is_train:
            self.image_files = sorted(glob(os.path.join(root, category, "train", "good", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの上半分を学習にし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))
                target_anomaly_type = ["good-" + path.split("/")[-2] for path in target_broken_files]
                self.image_files = self.image_files + target_broken_files[:len(target_broken_files)//2]
                self.anomaly_type = self.anomaly_type + target_anomaly_type[:len(target_anomaly_type)//2]

            if self.scenario == "N2A":
                # N2AのTrainの場合、学習データの疑似異常は省く
                self.image_files = [img_path for img_path in self.image_files if not "pseudo_" in img_path]
                self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

        else:
            self.image_files = sorted(glob(os.path.join(root, category, "test", "*", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの下半分をテストにし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))

                target_broken_files = target_broken_files[:len(target_broken_files)//2]
                image_files_temp = []
                anomaly_type_temp = []
                for img_file, ano_type in zip(self.image_files, self.anomaly_type):
                    if img_file in target_broken_files:
                        continue
                    if ano_type == self.broken_type:
                        ano_type = f"good-{ano_type}"
                    image_files_temp.append(img_file)
                    anomaly_type_temp.append(ano_type)

                self.image_files = image_files_temp
                self.anomaly_type = anomaly_type_temp

            if self.scenario == "N2A":
                # N2AのTestの場合、疑似異常を異常として扱う。何も変更しなくてよい。
                pass

            if self.scenario == "Normal":
                #N2A-NormalのTestの場合、疑似異常を正常として扱う。
                self.anomaly_type = [f"good-{anomaly}" if anomaly == "pseudo_anomaly" else anomaly for anomaly in self.anomaly_type]


            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )

        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        anomaly = self.anomaly_type[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            return anomaly, image
        else:
            if "good" in anomaly:
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                ).convert("L")
                target = self.target_transform(target)
            return anomaly, image, target

    def __len__(self):
        return len(self.image_files)




class VisADataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(".JPG", ".png")
                ).convert("L")
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)


# new size:384, 576
#no new size:336, 512
class MVTecDataset_Filter(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.image_files = glob(
            os.path.join(root, category, "train", "good", "*.png")
        )
        

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        return image, image_file

    def __len__(self):
        return len(self.image_files)


if __name__ == "__main__":
    train = False
    path = "/mnt/kamiya/dataset/MVtec_AD"
    path = "/mnt/kamiya/dataset/MVtec_AD-N2A"
    dataset = MVTecDataset_Scenario(path, "bottle", "Normal", None, 224, is_train=train)

    for data in dataset:
        if train:
            print(data[0])
            print(data[1].shape)
        else:
            print(data[0])
            print(data[1].shape)
            print(data[2].shape)
            print(data[2].sum())
        input()