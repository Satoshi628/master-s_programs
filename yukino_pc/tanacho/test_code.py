import json
import pandas as pd

with open("data/train_meta.json") as f:
    ano_info = json.load(f)

df = pd.DataFrame.from_dict(ano_info, orient='index')
print(df)