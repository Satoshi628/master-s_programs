import requests
import json

#フロア検索
# url = "https://api.dmm.com/affiliate/v3/FloorList?api_id=aR3vVKRB626V6ycmbHBz&affiliate_id=doshikori721-990&output=json"
offset = 1
url = f"https://api.dmm.com/affiliate/v3/ItemList?api_id=aR3vVKRB626V6ycmbHBz&affiliate_id=doshikori721-990&site=FANZA&service=digital&floor=videoa&hits=10&offset={offset}&sort=date&output=json"
r = requests.get(url)
FANZA_dict = json.loads(r.text)
result = FANZA_dict["result"]
print(list(result.keys()))
print(result)