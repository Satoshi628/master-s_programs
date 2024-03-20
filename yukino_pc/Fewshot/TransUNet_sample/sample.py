### import ###

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


### setting ###
config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
config_vit.n_classes = 2
config_vit.n_skip = 3
image_size = 224


### model create ###
if "R50-ViT-B_16".find('R50') != -1:
    config_vit.patches.grid = (int(image_size / 16), int(image_size / 16))
model = ViT_seg(config_vit, img_size=image_size, num_classes=config_vit.n_classes).cuda()
model.load_from(weights=np.load(config_vit.pretrained_path))


### e.g.###

output = model(input)
loss = criterion(output, target)



