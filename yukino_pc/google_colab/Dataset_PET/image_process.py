from PIL import Image
import os

#絶対パス
copy_moto_path = "../Dataset375x500large"
copy_saki_path = "../Dataset128x128"

moto_paths = [os.path.join(copy_moto_path,"PET",path) for path in os.listdir(copy_moto_path+"/PET")]
saki_paths = [os.path.join(copy_saki_path,"PET",path) for path in os.listdir(copy_saki_path+"/PET")]

for moto_path, saki_path in zip(moto_paths,saki_paths):
    image_moto_paths = os.listdir(moto_path)

    for img_path in image_moto_paths:
        img = Image.open(os.path.join(moto_path,img_path))
        img = img.resize((128, 128), Image.LANCZOS)
        img.save(os.path.join(saki_path,img_path))
