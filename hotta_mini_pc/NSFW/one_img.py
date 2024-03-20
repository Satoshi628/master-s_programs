import numpy as np
import cv2
from nsfw_detector import predict
import cv2

#1024枚  16s
# model = predict.load_model('./nsfw_mobilenet2.224x224.h5')
#1024枚  45s
model = predict.load_model('./nsfw.299x299.h5')

img = cv2.imread("image4.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (299, 299))
img = img/255.

# img = np.stack([img for _ in range(1024)])

output = predict.classify_nd(model, img[None])
print(output)