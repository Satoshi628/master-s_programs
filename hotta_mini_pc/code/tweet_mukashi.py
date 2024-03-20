import os
import sys
import re
import argparse
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary
from time import sleep
import pyautogui as pag
import pyperclip
import pandas as pd
import numpy as np

from edit_video import crop_video, delete_video
from utils import GENRE_DICT, AFI_ID_DICT
from KONOYONOOWARI import KONOYONOOWARI_HENSU

CLICK_TIME = 0.1
X_moji_LIMIT = 140

def get_args():
    parser = argparse.ArgumentParser(description="FANZAデータ収集プログラム(ジャンル別)")
    parser.add_argument("-g", "--genre", default="", help="Genre")
    parser.add_argument("-a", "--afi_name", default="", help="affiliate name")
    parser.add_argument("-i", "--id", default="", help="twitter ID")
    parser.add_argument("-p", "--password", default="", help="twitter password")
    parser.add_argument("-s", "--savepath", default="", help="save csv path")
    parser.add_argument("-c", "--cupon_parcent", type=float, default=0.3, help="cupon parcent")
    parser.add_argument("--no_actress", action='store_true')
    parser.add_argument("--no_long", action='store_true')
    parser.add_argument("--no_comment", action='store_true')
    args = parser.parse_args()
    
    return args


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


def twitter_login(driver, login_id, password):
    sleep(3)
    xpath_click(driver, "//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div[1]/div/div/div[3]/div[5]/a")
    sleep(2)
    pag.press("tab")
    pag.press("tab")
    pag.press("tab")
    sleep(1)
    pag.write(login_id)
    pag.press("enter")
    sleep(1)
    pag.write(password)
    pag.press("enter")
    sleep(3)

def tweet(driver, text, file_path, affi_path):
    space_position = [480, 270]
    text_space_position = [490, 360]

    driver.find_element_by_xpath("//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/div[2]/div[1]"\
                                "/div/div/div/div[2]/div[1]/div/div/div/div/div/div/div/div/div/div/label/div[1]"\
                                "/div/div/div/div/div/div[2]/div/div/div/div/span/br").send_keys(text)
    
    #メディア投稿ボタンが反応しないバグがあるため空白をクリック
    pag.click(*space_position)
    sleep(2)
    if file_path is not None:
        xpath_click(driver, "//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div[2]/div[1]/div/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[1]")
        sleep(2)
        pag.write(file_path)
        sleep(2)
        pag.press("enter")
        sleep(10)
    
    #追加ボタンクリック
    xpath_click(driver, "//*[@id=\"react-root\"]/div/div/div[2]/main/div/div/div/div/div/div[3]/div/div[2]/div[1]/div/div/div/div[2]/div[2]/div[2]/div/div/div[2]/a")
    sleep(2)
    #アフィリエイトURL追加
    if np.random.rand() < args.cupon_parcent:
        cupon_link = change_afiID("https://al.dmm.co.jp/?lurl=https%3A%2F%2Fwww.dmm.co.jp%2Fdigital%2F-%2Fwelcome-coupon%2F&af_id=doshikori721-001&ch=toolbar&ch_id=twitter \n", AFI_ID_DICT[args.afi_name])
        affi_tweet = "新規の方なら90%OFFのクーポンが使える!\n"+ cupon_link + "本編はこちらから⤵⤵⤵⤵⤵⤵⤵⤵⤵\n" + affi_path
    else:
        affi_tweet = "本編はこちらから⤵⤵⤵⤵⤵⤵⤵⤵⤵\n" + affi_path
    pyperclip.copy(affi_tweet)
    pag.hotkey('ctrl', 'v')
    sleep(2)
    pag.click(*text_space_position)
    sleep(10)
    #投稿ボタンクリック
    xpath_click(driver, "//*[@data-testid=\"tweetButton\"]")
    sleep(5)

def change_afiID(afi_link, afi_id):
    return afi_link.replace("doshikori721-001", afi_id)

def choice_video(df):
    
    use_min = np.min(df["use"].values)
    if args.no_long:
        flag = df['genre'].str.contains("セット商品") | df['genre'].str.contains("福袋") | df['genre'].str.contains("4時間以上作品") | df['genre'].str.contains("16間以上作品")
    else:
        flag = np.zeros(len(df), dtype=bool)
    #使用回数が最小かつ動画があるもの
    choice_idx = np.argmax(1. * (df["use"] == use_min)* ~flag * ~df["video_path"].isnull() * np.random.rand(len(df)))
    
    update_info(df, choice_idx)

    #レビューが存在しない場合はタイトルを表示
    if not isinstance((df.loc[choice_idx, "reviews"]), float):
        reviews = df.loc[choice_idx, "reviews"].replace("\n\n", "\n").split("@@@")
    else:
        reviews = [df.loc[choice_idx, "title"]]

    if not isinstance((df.loc[choice_idx, "special"]), float):
        coupon = df.loc[choice_idx, "special"].replace("【みんなのお気に入り70％OFFキャンペーン】","【期間限定70％OFF】").split("@@@")
    else:
        coupon = [""]
    
    if coupon[0] != "":
        coupon_temp = "".join([f"┊✨✨ {c} ✨✨┊\n" for c in coupon])
    else:
        coupon_temp = ""
    
    tags = []
    if not args.no_actress:
        if not isinstance(df.loc[choice_idx, "actress"], float):
            tags.extend(["#" + act for act in df.loc[choice_idx, "actress"].split(" ")])

    if not isinstance(df.loc[choice_idx, "genre"], float):
        tags.extend(["#" + Genre for Genre in df.loc[choice_idx, "genre"].replace("・","@@@").split("@@@") if Genre in KONOYONOOWARI_HENSU])
    else:
        tags.append("#" + args.genre)

    if len(tags) > 5:
        tags = np.random.choice(tags, 5, replace=False)
    hashtag = " ".join(tags)
    hashtag = clean_tag(hashtag)

    #ツイートする文を選ぶ
    count_OK = False
    for r in reviews:
        if len(r) < X_moji_LIMIT - len(coupon_temp) - len(hashtag) - len("\n"):
            count_OK = True
            comment = r
            break

    #いいレビューが見つからない場合はタイトルを表示
    if not count_OK:
        if len(df.loc[choice_idx, "title"]) < X_moji_LIMIT - len(coupon_temp) - len(hashtag) - len("\n"):
            comment = df.loc[choice_idx, "title"]
        else:
            comment = ""
    if args.no_comment:
        comment = ""

    tweet_comment = coupon_temp + comment + "\n" + hashtag
    video_path = df.loc[choice_idx, "video_path"]
    if isinstance(video_path, float):
        video_path = None
    afi_link = df.loc[choice_idx, "ad_url"]
    df.loc[choice_idx, "use"] += 1

    return tweet_comment, video_path, afi_link


def get_info(driver):
    
    special_text = []
    
    elem = driver.find_element_by_class_name("hreview")

    try:
        special_text.append(text_convert(elem.find_element_by_class_name("red").text))
    except:
        print(f"ERROR:red text")

    try:
        special_text.append(text_convert(elem.find_element_by_class_name("tx-hangaku").text))
    except:
        print(f"ERROR:hangaku text")
    

    "@@@".join(special_text)


def update_info(df, idx):
    
    options = Options()
    # linux上で動かすときは次の2つも実行してください。
    # options.add_argument('--headless')
    # options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(options=options)

    url = df.loc[idx, "page_url"]
    driver.get(url)
    sleep(1)
    
    #年齢確認
    xpath_click(driver, "//*[@id=\"dm-content\"]/main/div/div/div[2]")
    
    #クーポン表示削除
    xpath_click(driver, "//*[@id=\"campaign-popup-close\"]/img")

    #get cupon info
    special_text = []
    
    elem = driver.find_element_by_class_name("hreview")

    try:
        special_text.append(text_convert(elem.find_element_by_class_name("red").text))
    except:
        print(f"ERROR:red text")

    try:
        special_text.append(text_convert(elem.find_element_by_class_name("tx-hangaku").text))
    except:
        print(f"ERROR:hangaku text")
    

    df.loc[idx, "special"] = "@@@".join(special_text)
    driver.quit()

def clean_tag(hashtag):
    hashtag = re.sub(r'\([^()]*\)', '', hashtag)
    hashtag = hashtag.replace("#▼すべて表示する", "")
    return hashtag

args = get_args()
csv_path = args.savepath
print(csv_path)
if  os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, encoding="shift-jis")
else:
    df = None
    print("データがありません")
    sys.exit()

if len(df) == 0:
    print("ジャンルのデータがありません")
    sys.exit()

#動画情報など取得
tweet_comment, video_path, afi_link = choice_video(df)

#アフィリエイトIDを変更
afi_link = change_afiID(afi_link, AFI_ID_DICT[args.afi_name])

edit_video_path = "C:\\Users\\hotta_mini\\affi_data\\temp.mp4"
crop_video(video_path, edit_video_path)


options = Options()
# linux上で動かすときは次の2つも実行してください。
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')

driver = webdriver.Chrome(options=options)



#twitterに移動
driver.get('https://twitter.com/home?lang=ja')

#login
twitter_login(driver, args.id, args.password)

tweet(driver, tweet_comment, edit_video_path, afi_link)

driver.quit()

#編集された動画は削除
delete_video(edit_video_path)

try:
    df.to_csv(csv_path, encoding="shift-jis", index=False)
except Exception as e:
    print(f"ERROR:save csv")
    print(e)
    df.to_csv(csv_path, encoding="shift-jis", errors='ignore', index=False)