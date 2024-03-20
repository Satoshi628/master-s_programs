

with open("hiroyuki.txt", mode="r", encoding="utf-8") as f:
    text = f.read()
print(len(text))
text = text.replace("\n", "")
text = text.replace("。。", "")
print(len(text))

with open("hiroyuki_copy.txt", mode="w", encoding="utf-8") as f:
    f.write(text)