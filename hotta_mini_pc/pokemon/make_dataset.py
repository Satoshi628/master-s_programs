# coding: UTF-8
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import os
import urllib.request

def scraping():
    url = "http://games255.512.jp/pokewav_DL/index.html"
    data_url = "http://games255.512.jp"
    html = requests.get(url.format("1"))
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.content, features="html.parser")

    pokemon_dict1 = {"0"+item.text.replace(":",""): data_url + "/pokewav/" + item.text[:3]+ ".wav" for item in soup.find_all("td", class_="tdc1")}
    pokemon_dict2 = {"0"+item.text.replace(":",""): data_url + "/pokewav/" + item.text[:3]+ ".wav" for item in soup.find_all("td", class_="tdc2")}
    pokemon_dict = pokemon_dict1 | pokemon_dict2
    print(pokemon_dict)
    return pokemon_dict


def url2video(save_dir_path, audio_path):
    urllib.request.urlretrieve(audio_path, save_dir_path)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)



if __name__ == "__main__":
    pokemon_dict = scraping()

    poke_folder = "pokemon_audio"
    mkdir(poke_folder)
    for name, url in pokemon_dict.items():
        print(name)
        url2video(os.path.join(poke_folder, name + ".wav"), url)
    pass