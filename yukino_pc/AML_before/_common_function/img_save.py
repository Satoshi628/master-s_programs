import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as ttf
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


class Image():
    def __init__(self, color, model=None, max_image=64):
        self.__classification = len(color)
        self.__color = color
        self.__number_image = 0

        # ここより下中間層出力のための変数
        # modelクラスは=により共有される
        self.model = model
        self.__max_image = max_image
        self.__hiddenLayer_output = None
        self.__class_name = None

    def Segmentation_trans(self, inputs, mode='softmax'):
        if mode == 'softmax':
            inputs = F.softmax(inputs, dim=1)
            _, predicted = inputs.max(1)
        elif mode == 'sigmoid':
            inputs = torch.sigmoid(inputs)
            _, predicted = inputs.max(1)
        elif mode == 'linear':
            predicted = inputs
        # color=[class,b,H,W,color]
        color = predicted.new_tensor(self.__color).view(
            self.__classification, 1, 1, 1, 3)
        predicted = predicted.unsqueeze(-1).repeat(1, 1, 1, 3)
        predicted = torch.stack([(predicted == i) * color[i]
                                for i in range(self.__classification)])

        predicted = predicted.sum(dim=0).to(
            'cpu').detach().numpy().astype(np.uint8)

        return predicted

    def save_Segmentation(self, inputs, outputs, targets, folder='result', mode='softmax', inputs_mode='gray'):
        """Segmentation用画像保存

        Args:

        inputs (tensor): 入力画像

        outputs (tensor)): 出力画像

        targets (tensor): 正解ラベル

        folder (str, optional): 保存先フォルダ. Defaults to 'result'.

        mode (str, optional): 精度評価用活性化関数. Defaults to 'softmax'.

        inputs_mode (str, optional): 入力画像の色指定、gray以外だったらカラーにする. Defaults to 'gray'.
        """
        # データの画像化
        inputs = inputs.permute(0, 2, 3, 1) * 255
        inputs = inputs.to('cpu').detach().numpy().astype(np.uint8)
        outputs = self.Segmentation_trans(outputs, mode)
        targets = self.Segmentation_trans(targets, mode='linear')

        plt.clf()
        fig = plt.figure()
        plt.subplots_adjust(hspace=0.3)
        

        for i in range(inputs.shape[0]):
            title = fig.add_subplot(inputs.shape[0], 3, 3 * i + 1)
            if i == 0:
                title.set_title("image", fontsize=10,)
            plt.tick_params(labelbottom=False,
                            labelleft=False,
                            labelright=False,
                            labeltop=False,
                            bottom=False,
                            left=False,
                            right=False,
                            top=False)
            if inputs_mode == 'gray':
                plt.imshow(inputs[i, :, :, 0], cmap=inputs_mode)
            else:
                plt.imshow(inputs[i], cmap=inputs_mode)


            title = fig.add_subplot(inputs.shape[0], 3, 3 * i + 2)
            if i == 0:
                title.set_title("predicted", fontsize=10)
            plt.tick_params(labelbottom=False,
                            labelleft=False,
                            labelright=False,
                            labeltop=False,
                            bottom=False,
                            left=False,
                            right=False,
                            top=False)
            plt.imshow(outputs[i])


            title = fig.add_subplot(inputs.shape[0], 3, 3 * i + 3)
            if i == 0:
                title.set_title("label", fontsize=10)
            plt.tick_params(labelbottom=False,
                            labelleft=False,
                            labelright=False,
                            labeltop=False,
                            bottom=False,
                            left=False,
                            right=False,
                            top=False)
            plt.imshow(targets[i])

        plt.savefig('{}/img{:d}.png'.format(folder, self.__number_image))
        self.__number_image += 1
        plt.close()

    def save_HiddenLayer(self, module, inputs, mode='gray', out_index=None):
        """中間層出力

        Args:

        module (menber): 出力したい中間層名

        inputs (tensor): model入力する画像

        mode (str, optional): 画像の表示方法. Defaults to 'gray'.
        gray,color,heat,histから選べる。

        out_index (int, optional): 出力が複数の場合indexを指定する必要がある. Defaults to None.
        """
        assert self.model != None, "Image.save_HiddenLayer:どのモデルの中間層を出力するのか定義されていません"

        def forward_hock(self_module, inputs, outputs):
            self.__class_name = self_module.__class__.__name__
            if isinstance(outputs, list):
                assert out_index != None, "Image.save_HiddenLayer:出力が複数あるのにindexが指定されていません"
                self.__hiddenLayer_output = outputs[out_index].detach()
            else:
                self.__hiddenLayer_output = outputs.detach()

        handle = module.register_forward_hook(forward_hock)

        self.model.eval()
        self.model(inputs)

        handle.remove()

        # self.hiddenLayer_outputのconvの場合、大きさは[batch,c,W,H]
        # MLPの場合[batch,out]
        if len(self.__hiddenLayer_output.size()) == 4:
            for i in range(self.__hiddenLayer_output.size(0)):
                pass

        return


def print_image(img, file='test', mode='color'):
    """print_image
    2次元配列を画像として出力できる関数

    Args:

    img (list,numpy,tensor): printしたい二次元配列データ

    file (str, optional): Defaults to 'test'.
    保存するファイルの名前

    mode (str, optional): Defaults to 'color'.
    modeはcolor,gray,hist,heatから選べる
    """
    assert not type(img) in (str, int, float,
                             bool), "print_imageエラー:入力が配列ではありません"
    if isinstance(img, torch.Tensor):
        assert len(img.size()) == 2, "二次元Tensorではありません。"
        img = img.to(torch.float32)
        # [0,255]化
        img = 255*(img - img.min()) / (img.max() - img.min() + 1e-10)
        # numpy化
        img = img.to('cpu').detach().numpy().astype(np.uint8)
    elif isinstance(img, list):
        assert len(img) == 2, "二次元リストではありません。"
        # numpy化
        img = np.array(img).astype(np.float32)
        # [0,255]化
        img = 255*(img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
        # unsigned char化
        img = img.astype(np.uint8)
    elif isinstance(img, np.ndarray):
        assert len(img.shape) == 2, "二次元ndarrayではありません。"
        # float化
        img = img.astype(np.float32)
        # [0,255]化
        img = 255*(img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)
        # unsigned char化
        img = img.astype(np.uint8)

    if not isinstance(file, str):
        file = str(file)

    if mode is 'color':
        plt.imshow(img)
    if mode is 'gray':
        plt.imshow(img, cmap='gray')
    if mode is 'heat':
        plt.imshow(img, cmap='magma')
        plt.colorbar()
    if mode is 'hist':
        plt.hist(img, bins=256, normed=True)

    if not os.path.exists("print_image"):
        os.mkdir("print_image")
    # 画像保存
    plt.savefig('print_image/{}.png'.format(file))


def img_save_trim(device, inputs=None, outputs=None, label=None, trim=None, folder='result1', n_o_picture=0):
    trim = (trim != outputs) * 1.0
    outputs = F.softmax(outputs, dim=1)
    # 最大ベクトル
    _, predicted = outputs.max(1)
    number_of_img = inputs.size(0)
    color = torch.stack(
        [
            torch.full((256, 256), 3, dtype=torch.float),
            torch.full((256, 256), 2, dtype=torch.float),
            torch.full((256, 256), 1, dtype=torch.float),
        ]
    ).cuda(device)
    trim_color = torch.ones(number_of_img, 4, 3, 256, 256)
    color = color.repeat(number_of_img, 1, 1, 1)
    color = color.permute(0, 2, 3, 1)
    predicted = predicted.view(
        predicted.shape[0], predicted.shape[1], predicted.shape[2], 1)
    img = color.eq(predicted) * 255
    label = label.view(label.shape[0], label.shape[1], label.shape[2], 1)
    label = color.eq(label) * 255
    trim = trim*255
    trim = torch.unsqueeze(trim, -1)
    trim = trim.repeat(1, 1, 1, 1, 3)

    inputs = inputs.permute(0, 2, 3, 1)*255

    # numpyにする
    inputs_np = inputs.to('cpu').detach().numpy()
    img_np = img.to('cpu').detach().numpy()
    label_np = label.to('cpu').detach().numpy()
    trim_np = trim.to('cpu').detach().numpy()

    inputs_np = inputs_np.astype(np.uint8)
    img_np = img_np.astype(np.uint8)
    label_np = label_np.astype(np.uint8)
    trim_np = trim_np.astype(np.uint8)

    fig = plt.figure()
    plt.subplots_adjust(hspace=0.8)
    for i in range(number_of_img):
        title = fig.add_subplot(number_of_img, 7, 7*i + 1)
        title.set_title("image", fontsize=10)
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(inputs_np[i, :, :, 1], cmap='gray')

        title = fig.add_subplot(number_of_img, 7, 7*i + 2)
        title.set_title("predicted", fontsize=10)
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(img_np[i, :, :, :])

        title = fig.add_subplot(number_of_img, 7, 7*i + 3)
        title.set_title("trim[0]", fontsize=10)
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(trim_np[i, 0, :, :, :])

        title = fig.add_subplot(number_of_img, 7, 7*i + 4)
        title.set_title("trim[1]", fontsize=10)
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(trim_np[i, 1, :, :, :])

        title = fig.add_subplot(number_of_img, 7, 7*i + 5)
        title.set_title("trim[2]", fontsize=10)
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(trim_np[i, 2, :, :, :])

        title = fig.add_subplot(number_of_img, 7, 7*i + 6)
        title.set_title("trim[3]", fontsize=10)
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(trim_np[i, 3, :, :, :])

        title = fig.add_subplot(number_of_img, 7, 7*i + 7)
        title.set_title("Label", fontsize=10)
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        plt.imshow(label_np[i, :, :, :])
    plt.savefig('{}/img{:d}.png'.format(folder, n_o_picture))
