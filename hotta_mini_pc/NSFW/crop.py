import numpy as np
import os
from nsfw_detector import predict
import cv2

#1024枚  16s
# model = predict.load_model('./nsfw_mobilenet2.224x224.h5')
#1024枚  45s
model = predict.load_model('./nsfw.299x299.h5')


porn_ths = 0.20
# 出力フォルダ作成
os.makedirs("-20", exist_ok=True)

cap = cv2.VideoCapture("sqte431_mhb_w.mp4")
COUNT = 1000
count = COUNT
images = []
save_imgs = []
img_append = images.append
save_img_append = save_imgs.append

FPS = 10
fps_count = FPS
img_count = 0
while True:
    # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
    is_image, frame_img = cap.read()
    fps_count -= 1
    if fps_count != 0:
        continue
    if is_image:
        fps_count = FPS
        save_img_append(frame_img.copy())
        img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = img/255.
        img_append(img)
        count -= 1
        if count <= 0:
            images = np.stack(images)
            output = predict.classify_nd(model, images)

            for s_img, out_dict in zip(save_imgs, output):
                porn = out_dict["porn"]
                # sexy = out_dict["sexy"]
                if porn < porn_ths:
                    cv2.imwrite(f"-20\\{img_count:05}.png", s_img)
                    img_count += 1
            images = []
            save_imgs = []
            img_append = images.append
            save_img_append = save_imgs.append
            count = COUNT


    else:
        # フレーム画像が読込なかったら終了
        break

images = np.stack(images)
output = predict.classify_nd(model, images)

for s_img, out_dict in zip(save_imgs, output):
    porn = out_dict["porn"]
    # sexy = out_dict["sexy"]
    if porn < porn_ths:
        cv2.imwrite(f"-20\\{img_count:05}.png", s_img)
        img_count += 1