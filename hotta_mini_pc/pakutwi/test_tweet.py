import os
import sys
import re
import json
import shutil
import argparse
import requests
from datetime import datetime, date

from glob import glob
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import chromedriver_binary
from time import sleep
import pandas as pd
import numpy as np
from slack_sdk import WebClient
import cv2

from utils import COUPON_AFILINK_FANZA, GENRE_DICT, AFI_ID_DICT

CLICK_TIME = 0.1

def get_args():
    parser = argparse.ArgumentParser(description="ツイパクするカスみたいなコード")
    parser.add_argument("-a", "--affi_name", default="", help="affiliate name")
    parser.add_argument("-i", "--id", default="", help="twitter ID")
    parser.add_argument("-p", "--password", default="", help="twitter password")
    args = parser.parse_args()
    
    if not args.affi_name in AFI_ID_DICT:
        raise ValueError(f"Genreは{list(AFI_ID_DICT.keys())}から選択してください")
    return args



# サポート外の型が指定されたときの挙動を定義
def custom_default(o):
    if hasattr(o, '__iter__'):
        # イテラブルなものはリストに
        return list(o)
    elif isinstance(o, (datetime, date)):
        # 日時の場合はisoformatに
        return o.isoformat()
    else:
        # それ以外は文字列に
        return str(o)

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

def change_affiID(afi_link, afi_id):
    return afi_link.replace("doshikori721-001", afi_id)

class Paku_Tweet():
    def __init__(self, args):
        self.args = args
        self.driver = None
    
    def open_driver(self):
        options = Options()
        # options.binary_location = 'C:\\Users\\hotta_mini\\Downloads\\chrome-win64\\chrome-win64\\chrome.exe'

        # linux上で動かすときは次の2つも実行してください。
        # options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--start-maximized')
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'
        options.add_argument(f'--user-agent={user_agent}')
        self.driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
        self.wait = WebDriverWait(self.driver, 20)
    
    def close_driver(self):
        self.driver.quit()
        self.driver = None
    
    def FANZA_login(self):
        self.driver.get("https://accounts.dmm.co.jp/service/login/password")
        sleep(2)
        self.driver.find_element(By.XPATH, "//*[@id=\"login_id\"]").send_keys("a.affiliate0830@gmail.com")
        self.driver.find_element(By.XPATH, "//*[@id=\"password\"]").send_keys("meicraft")
        sleep(1)
        xpath_click(self.driver, "//*[@id=\"loginbutton_script_on\"]/span/input")
        sleep(1)

    def twitter_login(self):
        self.driver.get("https://twitter.com/i/flow/login")
        #ボタンが押せるようになるまで待機
        self.wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id=\"layers\"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[6]")))

        id_elem = self.driver.find_element(By.XPATH, "//*[@id=\"layers\"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input")
        id_elem.send_keys(self.args.id)
        sleep(0.2)
        xpath_click(self.driver, "//*[@id=\"layers\"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[6]")
        
        self.wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@data-testid=\"LoginForm_Login_Button\"]")))

        pw_elem = self.driver.find_element(By.XPATH, "//*[@id=\"layers\"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input")
        pw_elem.send_keys(self.args.password)
        sleep(0.2)
        xpath_click(self.driver, "//*[@data-testid=\"LoginForm_Login_Button\"]")
        sleep(3)
    
    def confirm_age(self):
        #年齢確認
        xpath_click(self.driver, "//*[@id=\"dm-content\"]/main/div/div/div[2]")
        sleep(1)
    
    def del_coupon(self):
        coupon_xpath = "//*[@id=\"react-popup-parts-root\"]/div/div/div/div[2]"
        if len(self.driver.find_elements(By.XPATH, coupon_xpath)):
            #クーポン表示削除
            xpath_click(self.driver, coupon_xpath)
            sleep(1)

    def get_affiliate_link(self, link):
        self.driver.get(link)

        try:
            self.wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id=\"react-popup-parts-root\"]/div/div/div/div[2]")))
        except:
            print("Time Out")
        self.del_coupon()

        xpath_click(self.driver, "//*[@id=\"affiliate-toolbar\"]/div[1]/ul/li[1]")

        sleep(0.5)
        #画像とテキストの物に変更
        xpath_click(self.driver, "//*[@id=\"affiliate-toolbar\"]/div[2]/div[2]/ul/li[3]")
        sleep(0.5)
        html = self.driver.find_element(By.CLASS_NAME, "c-contents__source").text
        url = re.findall('(?<=href=\").+?(?=\")', html)[0].replace("package_text_large", "twitter")
        sleep(0.25)
        class_click(self.driver, "c-icoClose")
        
        sleep(0.5)
        return url

    def paku_tweet(self, url):
        self.driver.get(url)
        sleep(4)
        self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        sleep(0.5)
        self.driver.execute_script('window.scrollTo(0, 0);')
        sleep(0.5)

        elem = self.driver.find_elements(By.XPATH, "//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/section/div/div/div[*]")
        #ツイートだけ残す
        elem = elem[:-1]
        elem = [e for idx, e in enumerate(elem) if idx != 1]


        file_name = "C:\\Users\\hotta_mini\\affi_data\\pakutwi\\images\\{}-{}.jpg"

        text_list = []
        images_path_list = []

        div_i = 1
        while len(self.driver.find_elements(By.XPATH, f"//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/section/div/div/div[{div_i}]")):
            elem = self.driver.find_elements(By.XPATH, f"//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/section/div/div/div[{div_i}]")
            #最後のツイートならおわり
            if len(self.driver.find_elements(By.XPATH, f"//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/section/div/div/div[{div_i+2}]")) == 0:
                break
            #div[2]は空白
            if div_i == 2:
                div_i += 1
                continue
            
            e = elem[0]

            if div_i == 1:
                text_elem = e.find_element(By.XPATH, "div/div/article/div/div/div[3]/div[1]")
            else:
                text_elem = e.find_element(By.XPATH, "div/div/article/div/div/div[2]/div[2]/div[2]")
            
            self.driver.execute_script("arguments[0].scrollIntoView();", e)
            text_list.append(text_elem.text)

            xpath_click(e, ".//*[@aria-label=\"画像\"]")
            sleep(1.5)

            n = 1
            save_path_list = []
            while True:
                if len(self.driver.find_elements(By.XPATH, "//*[@id=\"layers\"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div[1]/div[1]/div[1]/div/div/div/div/img")) != 0:
                    img_elem = self.driver.find_element(By.XPATH, "//*[@id=\"layers\"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div[1]/div[1]/div[1]/div/div/div/div/img")
                else:
                    img_elem = self.driver.find_element(By.XPATH, f"//*[@id=\"layers\"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div[1]/div[1]/div[1]/div[2]/div[1]/ul/li[{n}]/div/div/div/div/img")
                img_url = img_elem.get_attribute("src")
                ext = re.findall(r"format=(.*)&",img_url)[0]

                image = requests.get(img_url).content
                save_path = file_name.format(div_i, n)
                save_path_list.append(save_path)
                with open(save_path, "wb") as f:
                    f.write(image)
                
                n += 1
                if xpath_click(self.driver, "//*[@aria-label=\"次のスライド\"]"):
                    sleep(0.5)
                    continue
                else:
                    xpath_click(self.driver, "//*[@aria-label=\"閉じる\"]")
                    sleep(1)
                    break
            div_i += 1
            images_path_list.append(save_path_list)
        
        # try:
        #     elem[-1].find_element(By.XPATH, "div/div/article/div/div/div[2]/div[2]/div[2]").text
        # except:
        #     elem[-1].find_element(By.XPATH, "div/div/article/div/div/div[2]/div[3]/div[1]").text

        e = self.driver.find_elements(By.XPATH, "//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/section/div/div/div[*]")[-2]
        text_list.append(e.find_element(By.XPATH, "div/div/article/div/div/div[2]/div[2]/div[2]").text)
        fanza_url = e.find_element(By.XPATH, "div/div/article/div/div/div[2]/div[2]/div[3]//a").get_attribute("href")

        return text_list, images_path_list, fanza_url

    def tweet(self, texts, images_path, affiliate_link):
        #texts 漫画説明+affiliate説明 [N+1]
        #images_path 画像リスト [N]
        self.driver.get("https://twitter.com/compose/tweet")
        sleep(4)

        #テキスト入力
        text_xpath = "//*[@id=\"layers\"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div/div[3]/div[2]/div[1]"\
                    "/div/div/div/div[1]/div[2]/div/div/div/div/div/div/div/div/div/div/div/label/div[1]"\
                    "/div/div/div/div/div/div[2]/div/div/div/div/span/br"
        self.driver.find_element(By.XPATH, text_xpath).send_keys(texts[0])
        sleep(2)
        #メディア入力
        #複数の場合は\nで区切る
        self.driver.find_element(By.XPATH, "//input[@data-testid='fileInput']").send_keys("\n".join(images_path[0]))
        #ポストを追加
        n = 2
        for t, imgs in zip(texts[1:-1], images_path[1:]):
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@data-testid=\"addButton\"]")))

            xpath_click(self.driver, "//*[@data-testid=\"addButton\"]")
            sleep(1)
            self.driver.find_element(By.XPATH, "//*[@id=\"layers\"]/div[2]/div/div/div/div/div/div[2]/div[2]"\
                                            f"/div/div/div/div[3]/div[2]/div[{n}]/div/div/div/div[1]/div[2]"\
                                            "/div/div/div/div/div/div/div/div/div/div/div/label/div[1]"\
                                            "/div/div/div/div/div/div[2]/div/div/div/div/span/br").send_keys(t)

            self.driver.find_element(By.XPATH, "//input[@data-testid='fileInput']").send_keys("\n".join(imgs))
            n += 1

        #アフィリエイトポスト追加
        wait = WebDriverWait(self.driver, 20)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@data-testid=\"addButton\"]")))

        xpath_click(self.driver, "//*[@data-testid=\"addButton\"]")
        sleep(1)
        self.driver.find_element(By.XPATH, "//*[@id=\"layers\"]/div[2]/div/div/div/div/div/div[2]/div[2]"\
                                        f"/div/div/div/div[3]/div[2]/div[{n}]/div/div/div/div[1]/div[2]"\
                                        "/div/div/div/div/div/div/div/div/div/div/div/label/div[1]"\
                                        "/div/div/div/div/div/div[2]/div/div/div/div/span/br").send_keys("続きはこちらから⤵︎⤵︎⤵︎⤵︎⤵︎⤵︎⤵︎#ad" + f"\n{affiliate_link}")
        sleep(1)
        xpath_click(self.driver, "//*[@data-testid=\"tweetButton\"]")
        sleep(4)

    def upload_slack(self, text):
        slack_token = 'xoxb-436512348631-6120371602422-xJyE8PBnQqACtIhuaNgU9zpI'
        client = WebClient(slack_token)
        response = client.chat_postMessage(
            channel="D064E07342C",
            text=text
        )


    def __call__(self, Tweet_URL):
        print("driver open")
        self.open_driver()
        
        print("twitter login")
        try:
            self.twitter_login()
        except:
            self.upload_slack("ツイッターにログインできませんでした。\nどうにかしてください。")
            return 
        
        try:
            print("paku tweet")
            text_list, images_path_list, fanza_url = self.paku_tweet(Tweet_URL)
            print("Fanza login")
            self.FANZA_login()

            print("confirm age")
            self.confirm_age()

            print("affiliate bar")
            #アフィバー表示
            class_click(self.driver, "c-toolbar__open")
            sleep(0.5)

            print("get affiliate Link")
            affi_link = self.get_affiliate_link(fanza_url)
            affi_link = change_affiID(affi_link, AFI_ID_DICT[self.args.affi_name])
        except:
            self.upload_slack("異常なエラーです\n神谷に連絡してください")
            return 
        

        try:
            print("tweet")
            self.tweet(text_list, images_path_list, affi_link)

            self.upload_slack("ツイートしました")
        except:
            print("失敗")
            self.upload_slack("ツイート出来ませんでした")
        
        print("image delete")
        shutil.rmtree("C:\\Users\\hotta_mini\\affi_data\\pakutwi\\images")
        os.mkdir("C:\\Users\\hotta_mini\\affi_data\\pakutwi\\images")

        print("driver close")
        self.close_driver()

    def __del__(self):
        if self.driver is not None:
            self.driver.quit()


if __name__ == "__main__":
    args = get_args()
    paku = Paku_Tweet(args)
    bot_ID = "U063JAXHQCE"
    slack_channel = "D064E07342C"
    
    slack_token = 'xoxb-436512348631-6120371602422-xJyE8PBnQqACtIhuaNgU9zpI'
    Slack_client = WebClient(slack_token)

    time_stamp = Slack_client.conversations_history(channel=slack_channel)["messages"][0]['ts']
    while True:
        response = Slack_client.conversations_history(channel=slack_channel)["messages"][0]
        Bot_Flag = response['user'] != bot_ID
        TS_Flag = response['ts'] > time_stamp

        Text_Flag = "https://twitter.com/positive_Hero1/status/" in response['text']
        Text_Flag = True
        if TS_Flag:
            time_stamp = response['ts']
        if Bot_Flag & TS_Flag & ~Text_Flag:
            paku.upload_slack("リンクが正しくありません")
        if Bot_Flag & TS_Flag & Text_Flag:
            link_id = re.findall(r'\d+', response['text'])[-1]
            paku(f"https://twitter.com/positive_Hero1/status/{link_id}")
        
        sleep(5)