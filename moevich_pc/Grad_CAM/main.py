#coding: utf-8
##### ライブラリ読み込み #####
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import japanize_matplotlib
from torchvision.datasets.utils import download_url

############## main ##############
if __name__ == '__main__':
    # モデル設定
    device = torch.device('cuda:0')
    model = models.resnet50(pretrained=True, progress=True).cuda(device)
    model.eval()

    # ラベル辞書作製
    Keys = [i for i in range(1000)]

    if not Path("Image_net_Label.json").exists():
        # ファイルが存在しない場合はダウンロードする。
        download_url("https://git.io/JebAs", ".", "Image_net_Label.json")
    with open("Image_net_Label.json") as f:
        Values = json.load(f)
    Values = [item['ja'] for item in Values]
    """
    #両端の空白削除
    for index in range(len(Values)):
        Values[index] =Values[index].strip()
    """
    Label_dict = dict(zip(Keys, Values))

    # CAM定義
    cam = ScoreCAM(model=model, target_layer=model.layer4[2], use_cuda=device)

    # データ変換
    preprocess_image = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    preprocess_predict = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    # データロード、推論
    image_list = sorted(os.listdir("image"))
    print(image_list)
    for index, image_path in enumerate(image_list):
        # フォルダ内の画像の読み込み
        image = Image.open("image/{}".format(image_path)).convert('RGB')
        predict_img = preprocess_predict(image).cuda(device)
        predict_img = predict_img.unsqueeze(0)
        # 推論
        out = model(predict_img)
        out = F.softmax(out, dim=1)
        pre_value, pre_index = out.max(dim=1)

        # CAM作製
        # type(cam_img)=>ndarray
        cam_img = cam(input_tensor=predict_img,target_category=pre_index.item())
        # imageをndarrayにする。
        imaging_img = preprocess_image(image).permute(1, 2, 0)
        img_np = imaging_img.detach().numpy()
        visualization = show_cam_on_image(img_np, cam_img[0], use_rgb=True)
        plt.title("{}:{:.2f}%".format(
            Label_dict[pre_index.item()], pre_value.item() * 100))
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(visualization)
        if not os.path.exists("CAM_image"):
            os.mkdir("CAM_image")
        # 画像保存
        plt.savefig('CAM_image/{}_{}.png'.format(image_path,
                    cam.__class__.__name__))
