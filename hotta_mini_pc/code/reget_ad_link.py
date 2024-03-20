import os
import argparse
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
import chromedriver_binary
from time import sleep
import pyautogui as pag
import pyperclip
import pandas as pd

CLICK_TIME = 0.1

def get_args():
    parser = argparse.ArgumentParser(description="csvファイルの修正")
    parser.add_argument(
        "-f", "--file_path", default="", help="csv file path")
    args = parser.parse_args()
    
    return args

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
        elem = driver.find_element_by_class_name(class_name)
        elem.click()
        sleep(CLICK_TIME)
    except:
        print(f"ERROR:class_click {class_name}")
        return False
    return True

def xpath_click(driver, xpath):
    try:
        elem = driver.find_element_by_xpath(xpath)
        elem.click()
        sleep(CLICK_TIME)
    except:
        print(f"ERROR:xpath_click {xpath}")
        return False
    return True

def FANZA_login():
    tag_position = [900, 165]
    login_button_position = [630, 220]
    pag.click(*tag_position)
    pag.click(*login_button_position)
    sleep(1)
    pag.write("a.affiliate0830@gmail.com")
    pag.press("tab")
    pag.write("meicraft")
    pag.press("tab")
    pag.press("enter")
    sleep(2)

def get_ad_twitter_link(driver):
    link_bar_position = [360, 160]
    clipboard_save_position = [630, 950]
    background_position = [50, 500]
    afiID_bar_position = [440, 270]
    afiID_select_position = [0, 20]
    sleep(2)
    pag.click(*link_bar_position)
    sleep(1)
    url = xpath_click(driver, "//*[@id=\"affiliate-toolbar\"]/div[2]/div[2]/div[4]/div/button[2]")
    url = pyperclip.paste().replace("ch_id=link", "ch_id=twitter ")
    pyperclip.copy("")
    pag.click(*background_position)
    sleep(1)
    return text_convert(url)

args = get_args()
csv_path = args.file_path

if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, encoding="shift-jis")
else:
    df = None

options = Options()
# linux上で動かすときは次の2つも実行してください。
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')

driver = webdriver.Chrome(options=options)

url = "https://www.dmm.co.jp/digital/videoa/-/list/=/sort=ranking/page=1/"
driver.get(url)

#年齢確認
xpath_click(driver, "//*[@id=\"dm-content\"]/main/div/div/div[2]")

#ログイン
FANZA_login()

#アフィバー表示
class_click(driver, "c-toolbar__open")


for i in range(len(df)):
    url = df.loc[i, "page_url"]
    driver.get(url)
    sleep(1)
    
    #クーポン表示削除
    xpath_click(driver, "//*[@id=\"campaign-popup-close\"]/img")

    #広告リンク作成
    url = get_ad_twitter_link(driver)
    df.loc[i, "ad_url"] = url
    
    try:
        df.to_csv(csv_path, encoding="shift-jis", index=False)
    except Exception as e:
        print(f"ERROR:save csv")
        print(e)
        df.to_csv(csv_path, encoding="shift-jis", errors='ignore', index=False)
    sleep(1)

driver.quit()

# "//*[@id=\"main-ds\"]/div[2]/table[1]/tbody/tr/td[1]/div/p/a"
# "//*[@id=\"main-ds\"]/div[2]/table[1]/tbody/tr/td[1]/a"
# "//*[@id=\"main-ds\"]/div[2]/table[2]/tbody/tr/td[1]/a"
# "//*[@id=\"main-ds\"]/div[2]/table[1]/tbody/tr/td[2]/a"
