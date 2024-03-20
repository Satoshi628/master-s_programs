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
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
import chromedriver_binary
from time import sleep
import pyautogui as pag
import pyperclip
import pandas as pd
import cv2

from utils import GENRE_DICT


def get_args():
    parser = argparse.ArgumentParser(description="tiktok投稿")
    parser.add_argument("-i", "--id", default="", help="twitter ID")
    parser.add_argument("-p", "--password", default="", help="twitter password")
    parser.add_argument("-s", "--savepath", default="C:\\Users\\hotta_mini\\affi_data\\data.csv", help="save csv path")
    args = parser.parse_args()
    
    # if not args.genre in GENRE_DICT:
    #     raise ValueError(f"Genreは{list(GENRE_DICT.keys())}から選択してください")
    return args

#strからdatetimeに変換
#datetime.datetime.strptime(test, "%Y-%m-%d %H:%M:%S.%f")

def text_convert(string):
    string = string.replace(u"\uff3c",u"\u005c")
    string = string.replace(u"\uff5e",u"\u301c")
    string = string.replace(u"\u2225",u"\u2016")
    string = string.replace(u"\uff0d",u"\u2212")
    string = string.replace(u"\uffe0",u"\u00a2")
    string = string.replace(u"\uffe1",u"\u00a3")
    string = string.replace(u"\uffe2",u"\u00ac")
    return string

def class_click(driver, class_name):
    try:
        elem = driver.find_element(By.CLASS_NAME, class_name)
        elem.click()
        sleep(CLICK_TIME)
    except:
        print(f"ERROR:class_click {class_name}")
        return False
    return True

def xpath_click(driver, xpath):
    try:
        elem = driver.find_element(By.XPATH, xpath)
        elem.click()
        sleep(CLICK_TIME)
    except:
        print(f"ERROR:xpath_click {xpath}")
        return False
    return True

class TikTok():
    def __init__(self, args):
        self.args = args
        self.driver = None
    
    def open_driver(self):
        options = Options()
        # linux上で動かすときは次の2つも実行してください。
        # options.add_argument('--headless')
        # options.add_argument('--no-sandbox')
        options.add_argument('--start-maximized')
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'
        options.add_argument(f'--user-agent={user_agent}')
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)

    def close_driver(self):
        self.driver.quit()
        self.driver = None
    
    def open_csv(self):
        if  os.path.isfile(self.args.savepath):
            self.df = pd.read_csv(self.args.savepath, encoding="shift-jis")
            return True
        else:
            self.df = pd.DataFrame([])
            return False
    
    def save_csv(self):
        try:
            self.df.to_csv(self.args.savepath, encoding="shift-jis", index=False)
            return True
        except Exception as e:
            print(f"ERROR:save csv")
            print(e)
            self.df.to_csv(self.args.savepath, encoding="shift-jis", errors='ignore', index=False)
            return False
    
    def TikTok_login(self):
        self.driver.get("https://www.tiktok.com/login/phone-or-email/email")
        sleep(2)

        self.driver.find_element(By.XPATH, "//*[@id=\"loginContainer\"]/div[1]/form/div[1]/input").send_keys(self.args.id)
        self.driver.find_element(By.XPATH, "//*[@id=\"loginContainer\"]/div[1]/form/div[2]/div/input").send_keys(self.args.password)
        sleep(0.5)
        
        xpath_click(self.driver, "//*[@id=\"loginContainer\"]/div[1]/form/button")
        sleep(3)


    def __call__(self):
        print("csv open")
        self.open_csv()

        print("driver open")
        self.open_driver()

        print("TikTok login")
        self.TikTok_login()

        input()

    def __del__(self):
        if self.driver is not None:
            self.driver.quit()


if __name__ == "__main__":
    args = get_args()
    
    args = get_args()
    tik = TikTok(args)
    tik()