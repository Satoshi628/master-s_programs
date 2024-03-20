# http://api.e-stat.go.jp/rest/3.0/app/
# https://api.e-stat.go.jp/rest/3.0/app/json/getStatsList?appId="3658e525e5a555ff47e1c216273ba955e27a15a2"&lang=J&statsCode=00200211
import requests
import json
from io import BytesIO
from PIL import Image, ImageOps


# URLとクエリパラメータを設定
url = "https://pixel-image.vercel.app/api"
params = {
    "image": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    "size": "15",
    "k": "50"
}

# GETリクエストを送信
response = requests.get(url, params=params)
print(response.status_code)
img = Image.open(BytesIO(response.content))
img.show()

# # レスポンスをJSONファイルに保存
# data = response.json()  # レスポンスをJSON形式に変換
# with open("response.json", "w", encoding="utf-8") as json_file:
#     json.dump(data, json_file, ensure_ascii=False, indent=4)
# print("JSONファイルに保存しました。")
