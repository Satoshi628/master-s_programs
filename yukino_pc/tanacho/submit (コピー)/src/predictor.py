import os
import json
import glob
from PIL import Image
import torchvision.transforms as tf
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_changer(tensor, vector_dim, dim=-1, bool_=False):
    """index tensorをone hot vectorに変換する関数

    Args:
        tensor (torch.tensor,dtype=torch.long): index tensor
        vector_dim (int): one hot vectorの次元。index tensorの最大値以上の値でなくてはならない
        dim (int, optional): one hot vectorをどこの次元に組み込むか. Defaults to -1.
        bool_ (bool, optional): Trueにするとbool型になる。Falseの場合はtorch.float型. Defaults to False.

    Raises:
        TypeError: index tensor is not torch.long
        ValueError: index tensor is greater than vector_dim

    Returns:
        torch.tensor: one hot vector
    """
    if bool_:
        data_type = bool
    else:
        data_type = torch.float

    if tensor.dtype != torch.long:
        raise TypeError("入力テンソルがtorch.long型ではありません")
    if tensor.max() >= vector_dim:
        raise ValueError(f"入力テンソルのindex番号がvector_dimより大きくなっています\ntensor.max():{tensor.max()}")

    # one hot vector用単位行列
    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor]

    # one hot vectorの次元変更
    dim_change_list = list(range(tensor.dim()))
    # もし-1ならそのまま出力
    if dim == -1:
        return vector
    # もしdimがマイナスならスライス表記と同じ性質にする
    if dim < 0:
        dim += 1  # omsertは-1が最後から一つ手前

    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector


class ResNet(nn.Module):
    def __init__(self, resnet_type="resnet18", pretrained=True):
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            self.resnet_model = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet_model = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            self.resnet_model = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            self.resnet_model = torchvision.models.resnet152(pretrained=pretrained)

        # latest channel
        self.feature_layer = nn.Linear(self.resnet_model.layer4[-1].conv2.weight.shape[1], 512)

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)  # width, heightは1/4
        x = self.resnet_model.layer2(x)  # width, heightは1/8
        x = self.resnet_model.layer3(x)  # width, heightは1/16
        out = self.resnet_model.layer4(x)  # width, heightは1/32
        out = out.mean(dim=(-1, -2))

        feature = self.feature_layer(out)
        return feature


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, reference_path, reference_meta_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            reference_path (str): Path to the reference data.
            reference_meta_path (str): Path to the meta data.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        
        try:
            cls.model = ResNet(resnet_type="resnet18", pretrained=False)
            cls.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
            cls.reference = sorted(os.listdir(reference_path))
            image_paths = [glob.glob(os.path.join(reference_path, ref, "*.jpg")) for ref in cls.reference]
            image_paths = [path for paths in image_paths for path in paths]

            with open(reference_meta_path) as f:
                cls.reference_meta = json.load(f)
            
            cls.class_num, cls.vector = make_class_vector(cls.model, image_paths)
            return True
        except:
            return False

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input (str): path to the image you want to make inference from

        Returns:
            dict: Inference for the given input.
        """
        # load an image and get the file name
        image = read_image(input)
        sample_name = os.path.basename(input).split('.')[0]
        vector = model_run(cls.model, image)
        
        # make prediction
        prediction = predict2answer(vector, cls.class_num, cls.vector)

        # make output
        output = {sample_name: prediction}

        return output


def read_image(path):
    image = Image.open(path).convert("RGB").resize([512, 512], Image.LANCZOS)
    
    transform = tf.Compose([tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                            ])

    image = transform(image)
    return image


def make_class_vector(model, image_paths):
    vector_list = []
    class_num = []
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.cuda(device)
    model.eval()
    with torch.no_grad():
        for path in image_paths[:40]:
            image = read_image(path)
            image = image[None].cuda(device, non_blocking=True)
            vector = model(image)
            vector = F.normalize(vector, dim=1)
            vector = vector.to("cpu").detach()
            vector_list.append(vector)
            class_num.append(int(path.split("/")[-2]))
        
    class_num = torch.tensor(class_num)
    vector_list = torch.cat(vector_list, dim=0)
    return class_num, vector_list

def model_run(model, image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.cuda(device)
    model.eval()
    image = image[None].cuda(device, non_blocking=True)
    vector = model(image)
    vector = F.normalize(vector, dim=1)
    vector = vector.to("cpu").detach()
    return vector

def predict2answer(pre_vector, class_num, class_vector):
    cos_similar = (pre_vector * class_vector).sum(dim=-1, keepdim=True)

    uni_class = torch.unique(class_num)
    onehot_class = one_hot_changer(class_num, vector_dim=uni_class.shape[0])

    score = (cos_similar * onehot_class).sum(dim=0) / (onehot_class.sum(dim=0) + 1e-6)

    predict_num = torch.argsort(score, descending=True)[:10].tolist()
    predict_num = [str(pre_num).zfill(3) for pre_num in predict_num]

    return predict_num


def predict2answer(pre_vector, class_num, class_vector):
    cos_similar = (pre_vector * class_vector).sum(dim=-1)

    uni_class = torch.unique(class_num)
    class_count = torch.zeros(uni_class.shape)
    sort_idx = torch.argsort(cos_similar, descending=True)
    sort_class_num = class_num[sort_idx]

    for topk in sort_class_num[:10]:
        class_count[topk] += 1
    print(class_count)

    #一位を決める
    predict_num = [str(int(torch.argmax(class_count))).zfill(3)]
    print(predict_num)
    while sort_class_num.shape[0] != 0:
        sort_class_num = sort_class_num[sort_class_num != torch.argmax(class_count)]
        predict_num.append(str(int(sort_class_num[0])).zfill(3))
        sort_class_num = sort_class_num[sort_class_num != sort_class_num[0]]
        print(predict_num)
    return predict_num



############## main ##############
if __name__ == '__main__':
    ScoringService.get_model('../model', '/mnt/kamiya/code/tanacho/data/images', '/mnt/kamiya/code/tanacho/data/train_meta.json')
    ScoringService.predict('/mnt/kamiya/code/tanacho/data/images/000/0.jpg')
