import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from PIL import Image
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import faiss
from sklearn import metrics
from dataset import  TrainLoader, TestLoader
import glob
import re
import sample
import scipy.ndimage as ndimage
import timm
from function import print_model, timm_extract
###################################################################
def POOL1d(inputs, pool, dim):
    if pool == 'avg':
        pool = torch.nn.AdaptiveAvgPool1d(dim)
    if pool == 'max':
        pool = torch.nn.MaxPool1d(dim)
    B,C,H,W = inputs.shape

    inputs = inputs.reshape(B,C,H*W).permute(0,2,1)
    outputs = pool(inputs)
    outputs = outputs.reshape(B,H,W,dim).permute(0,3,1,2)
    return outputs

###################################################################
os.makedirs("results", exist_ok=True)

#### params ####
epochs = 100
batch_size = 1
device = torch.device("cuda:1" if torch.cuda.is_available else "cpu")
_CLASSNAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metalnut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
sample_rate = 0.1
sampler = sample.ApproximateGreedyCoresetSampler(sample_rate, device) # greedy
# sampler = "random" #  random
resize = 256
crop = 224
unfolder = nn.Unfold(3, 1, 1, 1)
mean_img = []
mean_pxl = []
################


#### model ####
# model = timm.create_model("poolformer_m48", pretrained=True)
model = timm.create_model("wide_resnet101_2", pretrained=True)
# model = timm.create_model("vit_base_patch16_224", pretrained=True)

model = model.to(device)
# print_model(model)
# summary(model,
#         input_size = (1, 3, 224, 224),
#         col_names = ["output_size", "num_params"],
#         )
# input()
targets1 = model.layer2[3].act3
targets2 = model.layer3[22].act3
# targets = model.norm
###############


for classname in _CLASSNAMES:
    os.makedirs(os.path.join("results", classname), exist_ok=True)
    #### dataloader ####
    traindata = TrainLoader(classname, resize=resize, crop=crop, transform=None)
    trainloader = DataLoader(traindata, batch_size=batch_size)
    testdata = TestLoader(classname, resize=resize, crop=crop, transform=None)
    testloader = DataLoader(testdata, batch_size=batch_size)
    ####################


    #### extract ####
    features_train = []
    features_test = []
    test_img_path = []

    for index, (img, img_path) in enumerate(tqdm(trainloader, leave=False, desc='feature_extraction(train)')):
        img = img.to(device)
        # print(model.__dict__["_modules"].keys()) # ['patch_embed', 'pos_drop', 'blocks', 'norm', 'pre_logits', 'head']
        # print(get_graph_node_names(model))
        # input()

        """ vit & poolformer(single) """
        # feature = timm_extract(model, targets, img)
        # feature = feature.to("cpu").detach().numpy()

        """" poolformer(double) """
        # feature1 = timm_extract(model, targets1, img)
        # feature2 = timm_extract(model, targets2, img)
        # upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # feature2 = upsample(feature2)
        # feature = torch.cat([feature1, feature2], 1)
        # feature = feature.to("cpu").detach().numpy()

        """" wideresnet50 """
        feature1 = timm_extract(model, targets1, img)
        feature2 = timm_extract(model, targets2, img)
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        unfolded_feature1 = unfolder(feature1)
        unfolded_feature2 = unfolder(feature2)
        unfolded_feature1 = unfolded_feature1.reshape(*feature1.shape[:2], -1, *feature1.shape[-2:])
        unfolded_feature2 = unfolded_feature2.reshape(*feature2.shape[:2], -1, *feature2.shape[-2:])
        unfolded_feature1 = unfolded_feature1.mean(2)
        unfolded_feature2 = unfolded_feature2.mean(2)
        unfolded_feature1 = POOL1d(unfolded_feature1, 'avg', unfolded_feature2.shape[1])
        unfolded_feature2 = upsample(unfolded_feature2)
        unfolded_feature = torch.cat([unfolded_feature1, unfolded_feature2], 1)
        unfolded_feature = POOL1d(unfolded_feature, 'avg', unfolded_feature2.shape[1])
        unfolded_feature = unfolded_feature.to("cpu").detach().numpy()

        features_train.append(unfolded_feature)

    if isinstance(sampler, sample.ApproximateGreedyCoresetSampler):
        features_train = np.concatenate(features_train, 0)

        """" vit """
        # features_train = features_train[:,1:,:]

        """" poolformer """
        # features_train = features_train.transpose(0,2,3,1)

        """" wideresnet50 """
        features_train = features_train.transpose(0,2,3,1)

        features_train = features_train.reshape(-1, features_train.shape[-1])
        features_train = sampler.run(features_train)
    else:
        features_train = random.sample(features_train, int(len(features_train)*sample_rate))
        features_train = np.concatenate(features_train,0)

        """" vit """
        # features_train = features_train[:,1:,:]
        
        """" poolformer """
        # features_train = features_train.transpose(0,2,3,1)

        """" wideresnet50 """
        features_train = features_train.transpose(0,2,3,1)

        features_train = features_train.reshape(-1, features_train.shape[-1])

    for index, (img, img_path) in enumerate(tqdm(testloader, leave=False, desc='feature_extraction(test)')):
        img_path = img_path[0].split("/")[-2:]
        img_path = "_".join(img_path)
        img = img.to(device)
        ##### extract ####
        """ vit & poolformer(single) """
        # feature = timm_extract(model, targets, img)
        # feature = feature.to("cpu").detach().numpy()

        """" poolformer(double) """
        # feature1 = timm_extract(model, targets1, img)
        # feature2 = timm_extract(model, targets2, img)
        # upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # feature2 = upsample(feature2)
        # feature = torch.cat([feature1, feature2], 1)
        # feature = feature.to("cpu").detach().numpy()

        """" wideresnet50 """
        feature1 = timm_extract(model, targets1, img)
        feature2 = timm_extract(model, targets2, img)
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        unfolded_feature1 = unfolder(feature1)
        unfolded_feature2 = unfolder(feature2)
        unfolded_feature1 = unfolded_feature1.reshape(*feature1.shape[:2], -1, *feature1.shape[-2:])
        unfolded_feature2 = unfolded_feature2.reshape(*feature2.shape[:2], -1, *feature2.shape[-2:])
        unfolded_feature1 = unfolded_feature1.mean(2)
        unfolded_feature2 = unfolded_feature2.mean(2)
        unfolded_feature1 = POOL1d(unfolded_feature1, 'avg', unfolded_feature2.shape[1])
        unfolded_feature2 = upsample(unfolded_feature2)
        unfolded_feature = torch.cat([unfolded_feature1, unfolded_feature2], 1)
        unfolded_feature = POOL1d(unfolded_feature, 'avg', unfolded_feature2.shape[1])
        unfolded_feature = unfolded_feature.to("cpu").detach().numpy()

        ####################
        features_test.append(unfolded_feature)
        test_img_path.append(img_path)

    features_test = np.concatenate(features_test,0)

    """ vit """
    # features_test = features_test[:,1:,:]  # default ViT
        
    """" poolformer """
    # features_test = features_test.transpose(0,2,3,1)
    # features_test = features_test.reshape(features_test.shape[0], -1, features_test.shape[-1])

    """ wideresnet50 """
    features_test = features_test.transpose(0,2,3,1)
    features_test = features_test.reshape(features_test.shape[0], -1, features_test.shape[-1])

    B, N, C = features_test.shape
    features_test = features_test.reshape(-1, features_test.shape[-1])
    #################


    #### nearest neighbor search ####
    d = features_train.shape[-1]
    k = 1
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    index.add(features_train)
    D, I = index.search(features_test, k)
    ##########################

    #### segmentation ####
    anoseg = D.reshape(B, int(N**(1/2)), int(N**(1/2)))
    anoseg = torch.from_numpy(anoseg.astype(np.float32)).clone()
    anoseg = transforms.functional.resize(anoseg, size=(img.shape[-2:]), interpolation=transforms.InterpolationMode.BILINEAR)

    anoseg = anoseg.numpy()

    for i in range(B):
        anoseg[i,:,:] = ndimage.gaussian_filter(anoseg[i,:,:], sigma=4)
    anosco = anoseg.reshape(B,-1).max(1)

    max = anosco.max()
    min = anosco.min()
    anosco = (anosco - min) / (max - min)
    
    max = anoseg.max()
    min = anoseg.min()
    anoseg = (anoseg - min) / (max - min)
    # anoseg = anoseg / max * anosco.reshape(B,1,1)

    for i, img_path in enumerate(tqdm(test_img_path, leave=False, desc='visualization')):
        seg = np.uint8(anoseg[i,:,:]*255)
        seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
        fig, ax = plt.subplots()
        ax.imshow(seg[:,:,[2,1,0]])
        ax.title.set_text("image_score : {:.3f}".format(anosco[i]))
        fig.savefig(os.path.join("results", classname, img_path))
        plt.clf()
        plt.close()
    ######################


    #### evaluation ####
    def crop_center(img, crop_width, crop_height):
        w, h = img.size
        return img.crop(((
                        (w-crop_width)//2,
                        (h-crop_height)//2,
                        (w+crop_width)//2,
                        (h+crop_height)//2,
                        )))

    test_lbl_path = sorted(glob.glob(os.path.join("mvtec", classname, "ground_truth", "*", "*.png")))
    test_img_path2 = sorted(glob.glob(os.path.join("mvtec", classname, "test", "*", "*.png")))
    test_img_path = sorted(glob.glob(os.path.join("mvtec", classname, "test", "*", "*.png"))) 
    for i, path in enumerate(test_img_path):
        test_img_path[i] = "_".join(path.split("/")[-2:])
    for i, path in enumerate(test_lbl_path):
        path = "_".join(path.split("/")[-2:])
        test_lbl_path[i] = re.sub("_mask", "", path)

    test_seg = []
    test_sco = []

    for i, ipath in enumerate(test_img_path):
        if ipath in test_lbl_path:
            ipath  = ipath.split("_")
            ipath[-1] = ipath[-1].split(".")
            ipath[-1] = ipath[-1][0] + "_mask." + ipath[-1][1]
            if len(ipath) == 2 :
                ipath = os.path.join(ipath[0], ipath[1])
            elif len(ipath) == 3 :
                ipath = os.path.join("_".join(ipath[:2]), ipath[-1])
            else :
                ipath = os.path.join("_".join(ipath[:3]), ipath[-1])
            seg = Image.open(os.path.join("mvtec", classname, "ground_truth", ipath))
            seg = seg.resize((resize, resize), Image.Resampling.NEAREST)
            seg = crop_center(seg, crop, crop)
            seg = np.array(seg)/255
        else : 
            seg = np.zeros((224,224))
        test_seg.append(seg)
    for seg in test_seg:
        if seg.max() == 1 :
            test_sco.append(1)
        else:
            test_sco.append(0)
    test_seg = np.stack(test_seg)
    test_sco = np.array(test_sco)
    image_score = metrics.roc_auc_score(test_sco, anosco)
    pixel_score = metrics.roc_auc_score(test_seg.reshape(-1), anoseg.reshape(-1))
    print("[{}]image:{:.3f}  pixel:{:.3f}".format(classname, image_score, pixel_score))
    mean_img.append(image_score)
    mean_pxl.append(pixel_score)

print("mean_image:{:.3f}  mean_pixel:{:.3f}".format(sum(mean_img)/len(mean_img), sum(mean_pxl)/len(mean_pxl)))
####################
