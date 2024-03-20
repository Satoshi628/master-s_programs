import os
import re
import sys
import argparse
import requests
import copy
import json
import shutil

import numpy as np
from glob import glob

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
import chromedriver_binary
from time import sleep
import pyautogui as pag
import pyperclip
import pandas as pd
import cv2

from utils import GENRE_DICT


def get_args():
    parser = argparse.ArgumentParser(description="tiktok投稿")
    # parser.add_argument("-g", "--genre", default="", help="Genre")
    parser.add_argument("-a", "--afi_name", default="", help="affiliate name")
    parser.add_argument("-i", "--id", default="", help="twitter ID")
    parser.add_argument("-p", "--password", default="", help="twitter password")
    parser.add_argument("--no_long", action='store_true')
    parser.add_argument("--link_tiktok", action='store_true')
    args = parser.parse_args()
    
    # if not args.genre in GENRE_DICT:
    #     raise ValueError(f"Genreは{list(GENRE_DICT.keys())}から選択してください")
    return args



def open_BlueStacks():
    BS_pos = [220, 1060]
    pag.click(*BS_pos)
    sleep(10)

def close_BlueStacks():
    close_pos1 = [1872,14]
    close_pos2 = [1076,500]
    pag.click(*close_pos1,1)
    sleep(1)
    pag.click(*close_pos2)


def opne_media():
    media_group_pos = [720, 315]
    media_pos = [1150, 430]
    folder_pos = [500, 200]
    pag.click(*media_group_pos)
    sleep(1)
    pag.click(*media_pos)
    sleep(1)
    pag.click(*folder_pos)
    sleep(1)

def close_media():
    close_ps = [1903, 873]
    pag.click(*close_ps)
    sleep(1)


def open_folder(image_path):
    BS_pos = [170, 1060]
    path_bar_pos = [490, 170]
    pag.click(*BS_pos)
    sleep(1)
    pag.hotkey("win", "left")
    sleep(1)
    #エクスプローラーを選択
    pag.click(*BS_pos)
    sleep(1)
    pag.click(*path_bar_pos)
    pag.write(image_path)
    pag.press("enter")
    sleep(1)

def close_folder():
    folder_space_ps = [210,11]
    close_ps = [940,14]
    # pag.click(*folder_space_ps)
    sleep(0.5)
    pag.click(*close_ps)
    sleep(1)


def import_images(image_path):
    opne_media()
    open_folder(image_path)

    leftup_file_pos = [220, 260]
    bulestacks_folder_pos = [1400, 460]
    pag.click(*leftup_file_pos)
    sleep(0.5)
    pag.hotkey("ctrl", "a")
    sleep(0.5)
    pag.moveTo(*leftup_file_pos)
    sleep(0.5)
    pag.dragTo(*bulestacks_folder_pos, 1)
    sleep(2)
    close_folder()
    sleep(2)
    close_media()
    sleep(2)

def crop_img(imgpath, save_path):
    #H,W,3
    img = cv2.imread(imgpath)
    top, bottom, left, right = np.random.randint(0, 10, [4]) 
    img = img[left:-right, top:-bottom]
    cv2.imwrite(save_path, img)
    return save_path

def delete_media_files():
    opne_media()
    leftup_folder_pos = [880,380]
    delete_pos = [1140,880]
    
    pag.click(*leftup_folder_pos)
    sleep(1)
    for _ in range(7):
        pag.click(*delete_pos)
        sleep(4)
    
    close_media()

#可愛い #美女 #女優 #おすすめ #続きはプロフ 
def delete_back_memory():
    back_tab_pos = [580, 14]
    delete_pos = [1420, 110]

    pag.click(*back_tab_pos)
    sleep(1)
    pag.click(*delete_pos)
    sleep(1)

def post_tiktok():
    tiktok_pos = [970, 316]
    post_pos = [1100, 900]
    media_pos = [1240, 830]

    leftup_media_pos = [995, 142]
    right_move_pos = [167, 0]
    down_move_pos = [0, 167]

    next_button_pos = [1200, 890]
    text_space_pos = [1000, 200]
    music_pos = [1090, 90]

    pag.click(*tiktok_pos)
    sleep(6)
    pag.doubleClick(*post_pos)
    sleep(6)
    pag.doubleClick(*media_pos)
    sleep(4)
    for i in range(6):
        clicl_pos = copy.copy(leftup_media_pos)
        clicl_pos[0] += i%3 * right_move_pos[0]
        clicl_pos[1] += i//3 * down_move_pos[1]

        pag.click(*clicl_pos)
        sleep(1)

    #次へボタン押す
    pag.click(*next_button_pos)
    sleep(4)

    #ToDo:曲を選べるようにする
    music_select_pos = [965, 546]
    space_pos = [1100, 300]
    dist = 44
    pag.click(*music_pos)
    sleep(2)
    music_select_pos[-1] += dist * np.random.randint(7)
    pag.click(*music_select_pos)
    pag.click(*space_pos)


    #次へ行き文章を打ち込む
    pag.click(*next_button_pos)
    sleep(4)
    pag.click(*text_space_pos)
    sleep(2)
    sleep(0.5)
    pag.hotkey('ctrl', 'v')
    sleep(0.5)
    pag.press('space')
    sleep(1)
    #投稿する
    # pag.click(*next_button_pos)
    sleep(5)

    close_ps = [1363, 871]
    pag.click(*close_ps)
    sleep(4)

def tiktok_reload():
    tiktok_pos = [970, 316]

    pag.click(*tiktok_pos)
    sleep(20)
    close_ps = [1363, 871]
    pag.click(*close_ps)
    sleep(3)



if __name__ == "__main__":
    args = get_args()
    
    save_images_path = "C:\\Users\\hotta_mini\\affi_data\\actress"
    save_temp_path = f"C:\\Users\\hotta_mini\\affi_data\\tiktok_temp\\{args.id}"
    if args.link_tiktok:
        if not os.path.isdir(save_temp_path):
            os.makedirs(save_temp_path)
        with open(f"C:\\Users\\hotta_mini\\affi_data\\temp\\{args.id}.json", mode="r", encoding='UTF-8') as f:
            data_dict = json.load(f)
        
        if not data_dict["tweet"]:
            sys.exit()

        product_num = data_dict["product_number"]
        img_paths = sorted(glob(os.path.join(f"C:\\Users\\hotta_mini\\affi_data\\tiktok_data\\{product_num}", "*")))

    temp_paths = [os.path.join(save_temp_path, os.path.basename(path)) for path in img_paths]
    print(img_paths)
    print(temp_paths)
    sleep(1)
    for post, temp in zip(img_paths, temp_paths):
        crop_img(post, temp)

    # csv_path = f"C:\\Users\\hotta_mini\\affi_data\\data_{GENRE_DICT[args.genre]}.csv"
    # csv_path = f"C:\\Users\\hotta_mini\\affi_data\\data_20.csv"
    
    # if  os.path.isfile(csv_path):
    #     df = pd.read_csv(csv_path, encoding="shift-jis")
    # else:
    #     df = None
    #     print("データがありません")
    #     sys.exit()
    
    image_path = "C:\\Users\\hotta_mini\\affi_data\\test_img"
    #BlueStacksを起動した後にコピーするとバグるため起動する前に投稿用文章をコピー
    
    text = "#可愛い #美女 #女優 #おすすめ #続きはプロフ"
    pyperclip.copy(text)

    open_BlueStacks()
    delete_back_memory()
    import_images(save_temp_path)
    tiktok_reload()
    post_tiktok()

    delete_media_files()
    shutil.rmtree(save_temp_path)

    close_BlueStacks()