import numpy as np
import os
from nsfw_detector import predict
import cv2
import pandas as pd


csv_path = f"C:\\Users\\hotta_mini\\affi_data\\data_20.csv"
df = pd.read_csv(csv_path, encoding="shift-jis")
df = df[~(df["video_path"].isnull() | df["ad_url"].isnull())]

#1024枚  16s
# model = predict.load_model('./nsfw_mobilenet2.224x224.h5')
#1024枚  45s
model = predict.load_model('./nsfw.299x299.h5')
model2 = predict.load_model('./nsfw_mobilenet2.224x224.h5')


porn_ths = 0.30
for idx in range(len(df)):
    try:
        video_path = df.loc[idx, "video_path"]
    except:
        continue
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    print(file_name)
    if os.path.isdir(f"crop3\\{file_name}"):
        continue

    os.makedirs(f"crop3\\{file_name}", exist_ok=True)
    os.makedirs(f"crop3\\{file_name}", exist_ok=True)


    # 出力フォルダ作成
    cap = cv2.VideoCapture(video_path)
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

                for s_img, img, out_dict in zip(save_imgs, images, output):
                    porn = out_dict["porn"]
                    # sexy = out_dict["sexy"]
                    if porn < porn_ths:
                        img = cv2.resize(img, (224, 224))
                        out = predict.classify_nd(model, img[None])
                        out = out[0]["porn"]
                        if out < porn_ths:
                            cv2.imwrite(f"crop3\\{file_name}\\{img_count:05}.png", s_img)
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

    for s_img, img, out_dict in zip(save_imgs, images, output):
        porn = out_dict["porn"]
        # sexy = out_dict["sexy"]
        if porn < porn_ths:
            img = cv2.resize(img, (224, 224))
            out = predict.classify_nd(model2, img[None])
            out = out[0]["porn"]

            if out < porn_ths:
                cv2.imwrite(f"crop3\\{file_name}\\{img_count:05}.png", s_img)
            img_count += 1