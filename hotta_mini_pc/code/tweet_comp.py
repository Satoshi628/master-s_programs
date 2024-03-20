import os
import sys
import re
import json
import argparse
import requests
from datetime import datetime, date

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import pandas as pd
import numpy as np

from edit_video import crop_video, delete_video
from utils import COUPON_AFILINK_FANZA, GENRE_DICT, AFI_ID_DICT
from KONOYONOOWARI import KONOYONOOWARI_HENSU
from search_FANZA_comp import FANZA_Data_Collector

CLICK_TIME = 0.1
X_moji_LIMIT = 140

def get_args():
    parser = argparse.ArgumentParser(description="全体データベースからフィルタリングしてtweetする")
    parser.add_argument("-g", "--genre", default=[""], required=True, nargs="*", help="Genre")
    parser.add_argument("-a", "--afi_name", default="", help="affiliate name")
    parser.add_argument("-i", "--id", default="", help="twitter ID")
    parser.add_argument("-p", "--password", default="", help="twitter password")
    parser.add_argument("-s", "--savepath", default="", help="save csv path")
    parser.add_argument("-c", "--cupon_parcent", type=float, default=0.3, help="cupon parcent")
    parser.add_argument("--no_actress", action='store_true')
    parser.add_argument("--no_long", action='store_true')
    parser.add_argument("--no_comment", action='store_true')
    parser.add_argument("--link_tiktok", action='store_true')
    args = parser.parse_args()
    
    if not args.afi_name in AFI_ID_DICT:
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

def change_afiID(afi_link, afi_id):
    return afi_link.replace("doshikori721-001", afi_id)

def clean_tag(hashtag):
    hashtag = re.sub(r'\([^()]*\)', '', hashtag)
    hashtag = hashtag.replace("#▼すべて表示する", "")
    return hashtag


class Twitter(FANZA_Data_Collector):
    def __init__(self, args):
        super().__init__(args)

    
    def open_driver(self):
        options = Options()
        # linux上で動かすときは次の2つも実行してください。
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--start-maximized')
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'
        options.add_argument(f'--user-agent={user_agent}')
        self.driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
        self.wait = WebDriverWait(self.driver, 20)
    
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
    
    def tweet(self, text, video_path, add_text):
        #テキスト入力
        text_xpath = "//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/div[2]/div[1]"\
                    "/div/div/div/div[2]/div[1]/div/div/div/div/div/div/div/div/div/div/label/div[1]"\
                    "/div/div/div/div/div/div[2]/div/div/div/div/span/br"
        self.driver.find_element(By.XPATH, text_xpath).send_keys(text)
        sleep(2)
        #メディア入力
        #複数の場合は\nで区切る
        self.driver.find_element(By.XPATH, "//input[@data-testid='fileInput']").send_keys(video_path)

        #ポストを追加
        wait = WebDriverWait(self.driver, 20)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@data-testid=\"addButton\"]")))

        xpath_click(self.driver, "//*[@data-testid=\"addButton\"]")
        sleep(1)
        self.driver.find_element(By.XPATH, "//*[@id=\"layers\"]/div[2]/div/div/div/div/div/div[2]/div[2]"\
                                        "/div/div/div/div[3]/div[2]/div[2]/div/div/div/div[1]/div[2]"\
                                        "/div/div/div/div/div/div/div/div/div/div/div/label/div[1]"\
                                        "/div/div/div/div/div/div[2]/div/div/div/div/span/br").send_keys(add_text)
        sleep(20)
        xpath_click(self.driver, "//*[@data-testid=\"tweetButton\"]")
        sleep(1)

    def choice_df(self):
        # 長い動画フィルター
        if args.no_long:
            long_flag = self.df['genre'].str.contains("セット商品") | self.df['genre'].str.contains("福袋") | self.df['genre'].str.contains("4時間以上作品") | self.df['genre'].str.contains("16間以上作品")
        else:
            long_flag = np.zeros(len(self.df), dtype=bool)
        long_flag = ~long_flag

        #ジャンルフィルター
        genre = self.args.genre[::2]
        operator = self.args.genre[1::2]
        
        genre_flag = self.df['genre'].str.contains(genre[0])
        for g, ope in zip(genre[1:], operator):
            if ope == "and":
                genre_flag = eval("genre_flag & self.df[\'genre\'].str.contains(g)")
            elif ope == "or":
                genre_flag = eval("genre_flag | self.df[\'genre\'].str.contains(g)")
            else:
                raise ValueError("operatorはandかorを選択してください")

        flag = genre_flag * long_flag
        df = self.df[flag]

        use_min = np.min(df["use"].values)

        #使用回数が最小かつ動画、アフィリンクがあるもの
        choice_idx = np.argmax((df["use"] == use_min) * ~df["video_path"].isnull() * ~df["ad_url"].isnull() * np.random.rand(len(df)))
        
        prod_num = df.iloc[choice_idx]["product_number"]
        return prod_num

    def update_df(self, product_number):
        
        print("Fanza login")
        self.FANZA_login()

        print("confirm age")
        self.confirm_age()

        print("affiliate bar")
        #アフィバー表示
        class_click(self.driver, "c-toolbar__open")
        sleep(0.5)

        url = self.df.loc[self.df["product_number"] == product_number, "page_url"].values[0]
        self.driver.get(url)
        sleep(1)
        
        new_data = self.get_product()
        for k, v in new_data.items():
            if v:
                self.df.loc[self.df["product_number"] == product_number, k] = v

        self.df.loc[self.df["product_number"] == product_number, "use"] += 1

        return new_data

    def make_tweet(self, data):
        if self.args.no_comment:
            reviews = [""]
        elif data["reviews"] != "":
            reviews = data["reviews"].replace("\n\n", "\n").split("@@@")
        else:
            #レビューが存在しない場合はタイトルを表示
            reviews = [data["title"]]

        if data["sale"] != "":
            sale = "期間限定セール!今なら{}円".format(int(data["price"]))
        else:
            sale = ""
        
        if sale != "":
            sale = f"┊✨ {sale} ✨┊\n"
        else:
            sale = ""
        
        tags = []
        if not self.args.no_actress:
            if data["actress"] is not None:
                #'(?<=（).+?(?=）)'かっこが消せない
                actress = re.sub('（[^()]*）', "", data["actress"])
                tags.extend(["#" + act for act in actress.split(" ")])

        if data["genre"] != "":
            tags.extend(["#" + Genre for Genre in data["genre"].replace("・","@@@").split("@@@") if Genre in KONOYONOOWARI_HENSU])
        else:
            tags.extend(["#" + g for g in self.args.genre[::2]])

        if len(tags) > 5:
            tags = np.random.choice(tags, 5, replace=False)
        hashtag = " ".join(tags)
        hashtag = clean_tag(hashtag)

        #ツイートする文を選ぶ
        count_OK = False
        for r in reviews:
            if len(r) < X_moji_LIMIT - len(sale) - len(hashtag) - len("\n"):
                count_OK = True
                comment = r
                break

        #いいレビューが見つからない場合はタイトルを表示
        if not count_OK:
            if len(data["title"]) < X_moji_LIMIT - len(sale) - len(hashtag) - len("\n"):
                comment = data["title"]
            else:
                comment = ""
        if comment != "":
            comment = comment + "\n"
        
        tweet_comment = sale + comment + hashtag + " "


        video_path = data["video_path"]
        edit_video_path = os.path.join("C:\\Users\\hotta_mini\\affi_data\\temp", os.path.basename(video_path))
        crop_video(video_path, edit_video_path)

        
        afi_link = data["ad_url"]
        afi_link = change_afiID(afi_link, AFI_ID_DICT[self.args.afi_name])
        #アフィリエイトURL追加
        if np.random.rand() < self.args.cupon_parcent:
            coupon_link = change_afiID(COUPON_AFILINK_FANZA, AFI_ID_DICT[self.args.afi_name])
            moto_price = int(data["price"])
            ima_price = max(int(data["price"]-500), 0)
            add_comment = "新規の方なら500円OFFのクーポンが使える!\n"
            if ima_price == 0:
                add_comment = add_comment + "この作品がなんと無料!\n" + coupon_link + "本編はこちらから⤵⤵⤵⤵⤵⤵⤵⤵⤵\n"
            else:
                add_comment = add_comment + f"{moto_price}円から{ima_price}円に!\n" + coupon_link + "本編はこちらから⤵⤵⤵⤵⤵⤵⤵⤵⤵\n"
        else:
            add_comment = "本編はこちらから⤵⤵⤵⤵⤵⤵⤵⤵⤵\n" + afi_link
        
        sleep(1)
        return tweet_comment, edit_video_path, add_comment

    def __call__(self):
        print("csv open")
        self.open_csv()

        print("driver open")
        self.open_driver()
        
        print("choice dataframe")
        prod_num = self.choice_df()

        print("update dataframe")
        data = self.update_df(prod_num)

        if self.args.link_tiktok:
            saving_data = {"product_number": data["product_number"]}
            with open(f"C:\\Users\\hotta_mini\\affi_data\\temp\\{self.args.id}.json", mode="w", encoding='UTF-8') as f:
                json.dump(saving_data, f, indent=2, ensure_ascii=False, default=custom_default)

        print("make tweet")
        comment, media, add_comment = self.make_tweet(data)

        print("twitter login")
        self.twitter_login()
        
        print("tweet")
        # self.tweet("comment", "C:\\Users\\hotta_mini\\affi_data\\temp\\hmn00475.mp4", "add_comment")
        self.tweet(comment, media, add_comment)
        
        print("csv save")
        self.save_csv()

        print("delete temp video")
        #編集された動画は削除
        delete_video(media)

        print("driver close")
        self.close_driver()


if __name__ == "__main__":
    args = get_args()
    tweet = Twitter(args)
    tweet()
