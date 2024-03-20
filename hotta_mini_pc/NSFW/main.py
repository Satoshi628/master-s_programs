import numpy as np
import cv2
from nsfw_detector import predict
import cv2

#1024枚  16s
# model = predict.load_model('./nsfw_mobilenet2.224x224.h5')
#1024枚  45s
model = predict.load_model('./nsfw.299x299.h5')

# img = cv2.imread("IMG_5042.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (299, 299))
# img = img/255.

# # img = np.stack([img for _ in range(1024)])

# output = predict.classify_nd(model, img[None])
# print(output)

result_dict = {'drawings': [],
              'hentai': [],
              'neutral': [],
              'porn': [],
              'sexy': []}
class_value = {'drawings': 0.,
              'hentai': 1.,
              'neutral': 0.,
              'porn': 0.,
              'sexy': 0.5}

cap = cv2.VideoCapture("sqte431_mhb_w.mp4")
# cap.set(cv2.CAP_PROP_FPS, 10)
COUNT = 1000
count = COUNT
images = []
img_append = images.append

WH = [299, 299]
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
writer = cv2.VideoWriter('ero_detection.mp4', fmt, cap.get(cv2.CAP_PROP_FPS), WH)

# fps_count = 6
while True:
    # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
    is_image, frame_img = cap.read()

    if is_image:
        save_img = frame_img.copy()
        img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = img/255.
        img_append(img)
        count -= 1
        if count <= 0:
            images = np.stack(images)
            output = predict.classify_nd(model, images)

            for img, out_dict in zip(images, output):
                for k, v in out_dict.items():
                    result_dict[k].append(v)
                porn = out_dict["porn"]
                sexy = out_dict["sexy"]
                img = (img.copy()*255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.putText(img, f"{porn:.2%}, {sexy:.2%}", (0, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                writer.write(img)
            images = []
            img_append = images.append
            count = COUNT


    else:
        # フレーム画像が読込なかったら終了
        break

images = np.stack(images)
output = predict.classify_nd(model, images)

for img, out_dict in zip(images, output):
    for k, v in out_dict.items():
        result_dict[k].append(v)
    porn = out_dict["porn"]
    sexy = out_dict["sexy"]
    img = (img.copy()*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.putText(img, f"{porn:.2%}, {sexy:.2%}", (0, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
    writer.write(img)
#ファイルを閉じる
writer.release()