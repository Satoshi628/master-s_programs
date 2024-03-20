import os
import sys
import re
import argparse
import requests
import urllib.parse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
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
    parser = argparse.ArgumentParser(description="FANZAからセール品を収集してtwitterに投稿")
    parser.add_argument("-s", "--search", default="", help="search")
    parser.add_argument("-a", "--afi_name", default="", help="affiliate name")
    parser.add_argument("-i", "--id", default="", help="twitter ID")
    parser.add_argument("-p", "--password", default="", help="twitter password")
    parser.add_argument("--no_long", action='store_true')
    parser.add_argument("--no_comment", action='store_true')
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
    if np.random.rand() < 0.3:
        cupon_link = change_afiID("https://al.dmm.co.jp/?lurl=https%3A%2F%2Fwww.dmm.co.jp%2Fdigital%2F-%2Fwelcome-coupon%2F&af_id=doshikori721-001&ch=toolbar&ch_id=twitter \n", AFI_ID_DICT[args.afi_name])
        affi_tweet = "新規の方なら90OFFのクーポンが使える!\n"+ cupon_link + "本編はこちらから⤵⤵⤵⤵⤵⤵⤵⤵⤵\n" + affi_path
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
    
    #使用回数が最小かつ動画があるものかつセール検索をしたもの
    choice_idx = update_info_and_choice(df)

    #レビューが存在しない場合はタイトルを表示
    if not isinstance((df.loc[choice_idx, "reviews"]), float):
        reviews = df.loc[choice_idx, "reviews"].replace("\n\n", "\n").split("@@@")
    else:
        reviews = [df.loc[choice_idx, "title"]]

    if not isinstance((df.loc[choice_idx, "special"]), float):
        coupon = [s + "キャンペーン" for s in re.findall(r"\D％OFF", df.loc[choice_idx, "special"])]
    else:
        coupon = []
    
    if coupon:
        coupon_temp = "".join([f"┊✨✨ {c} ✨✨┊\n" for c in coupon])
    else:
        coupon_temp = ""
    
    tags = []
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

def video_download():
    sleep(20)
    video_position = [500,500]
    option_relative_position = [90,140]
    file_bar_position = [320,450]
    pag.click(*video_position, button="right")
    pag.move(*option_relative_position)
    pag.click()
    video_url = pyperclip.paste()
    
    
    result = False
    if video_url == "":
        return result, None, None
    
    file_name = video_url.split("/")[-2]
    save_path = f"C:\\Users\\hotta_mini\\affi_data\\data\\{file_name}.mp4"
    
    for i in range(5):
        try:
            r = requests.get(video_url, stream=True, timeout=10)
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(r.content)

        except requests.exceptions.SSLError:
            print('***** SSL エラー')
            break  # リトライしない
        except requests.exceptions.RequestException as e:
            print(f'***** requests エラー({e}): {i + 1}/{RETRY_NUM}')
            time.sleep(1)
        else:
            result = True
            break  # try成功
    pyperclip.copy("")
    return result, video_url, save_path

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

def select_hyouka(driver):
    elem = driver.find_element_by_xpath('//*[@id=\"review\"]/div[2]/div/div[3]/div[1]/div[1]/div/select')
    select = Select(elem)
    select.select_by_index(2)

def get_review(driver):
    #評価が高い順を選択
    try:
        elem = driver.find_element_by_xpath('//*[@id=\"review\"]/div[2]/div/div[3]/div[1]/div[1]/div/select')
    except:
        return {"reviews": ""}
    select = Select(elem)
    select.select_by_index(2)
    sleep(1)
    
    #レビュー表示ボタンをすべて押す
    elems = driver.find_elements_by_class_name("d-modtogglelink-open")
    for e in elems:
        e.click()
    
    #ページ最上にスクロール
    sleep(1)
    driver.execute_script('window.scrollTo(0, 0);')
    
    reviews = []
    elems = driver.find_elements_by_class_name("d-review__unit")
    for e in elems:
        if len(e.find_elements_by_class_name("d-rating-50")) > 0 or len(e.find_elements_by_class_name("d-rating-40")) > 0:
            text = e.find_element_by_class_name("d-review__unit__comment").text
            reviews.append(text_convert(text))
    # elems = driver.find_elements_by_class_name("d-review__unit__comment")
    # reviews = [text_convert(e.text) for e in elems]
    return {"reviews": "@@@".join(reviews)}

def get_info(driver):
    
    try:
        actress = text_convert(driver.find_element_by_id("performer").text)
    except:
        print(f"ERROR:performer text")
        actress = None
    
    special_text = []
    
    elem = driver.find_element_by_class_name("hreview")

    try:
        title = text_convert(elem.find_element_by_id("title").text)
    except:
        print(f"ERROR:title text")
        title = None

    #現在のURLを保存
    current_URL = driver.current_url

    try:
        special_text.append(text_convert(elem.find_element_by_class_name("red").text))
    except:
        print(f"ERROR:red text")

    try:
        special_text.append(text_convert(elem.find_element_by_class_name("tx-hangaku").text))
    except:
        print(f"ERROR:hangaku text")
    
    #ジャンル
    Genres = driver.find_element_by_xpath("//*[@id=\"mu\"]/div/table/tbody/tr/td[1]/table/tbody/tr[11]/td[2]").text
    Genres =  Genres.replace("  ", "@@@")

    return {"title": title, "page_url": current_URL, "actress": actress, "special": "@@@".join(special_text), "genre": Genres}


def get_sample(driver, elems, df):
    
    dataframe_list = []

    for e in elems:
        data = dict()
        #作品一つクリック
        e.find_element_by_class_name("txt").click()

        #クーポン表示削除
        xpath_click(driver, "//*[@id=\"campaign-popup-close\"]/img")
        sleep(1)

        #サンプル動画表示クリック
        sample_flag = xpath_click(driver, "//*[@id=\"detail-sample-movie\"]/div/a")
        #情報取得
        info = get_info(driver)
        
        data.update(info)
        data.update(get_review(driver))

        #ダウンロード
        if sample_flag:
            video_flag, video_url, save_path  = video_download()
            data["video_url"] = video_url
            if video_flag:
                data["video_path"] = save_path
            else:
                driver.back()
                sleep(1)
                continue
        else:
            driver.back()
            sleep(1)
            continue

        #広告リンク作成
        url = get_ad_twitter_link(driver)
        data["ad_url"] = url
        data["use"] = 0
        dataframe_list.append(data)
        driver.back()
        sleep(1)

        #保存
        new_df = pd.DataFrame(dataframe_list)

        df = pd.concat([df, new_df])

        #titleの重複対策
        df = df[~df.duplicated(subset='title')]
        
        try:
            df.to_csv(csv_path, encoding="shift-jis", index=False)
        except Exception as e:
            print(f"ERROR:save csv")
            print(e)
            df.to_csv(csv_path, encoding="shift-jis", errors='ignore', index=False)
        df = pd.read_csv(csv_path, encoding="shift-jis")

        return df

def update_info_and_choice(df):
    #class="tmb"
    serch_query = args.search + " -セット商品 -福袋 -4時間以上作品 -16間以上作品" if args.no_long else args.search
    serch_query = urllib.parse.quote(serch_query)

    url = f"https://www.dmm.co.jp/digital/videoa/-/list/search/=/?searchstr={serch_query}"
    options = Options()
    # linux上で動かすときは次の2つも実行してください。
    # options.add_argument('--headless')
    # options.add_argument('--no-sandbox')

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    sleep(1)
    
    #年齢確認
    xpath_click(driver, "//*[@id=\"dm-content\"]/main/div/div/div[2]")
    sleep(1)
    
    #ログイン
    FANZA_login()

    #アフィバー表示
    class_click(driver, "c-toolbar__open")
    sleep(1)
    elems = driver.find_elements_by_xpath("//*[@id=\"list\"]/li[*]")

    elems = [e for e in elems if len(e.find_elements_by_class_name("info-countdown-name")) != 0]
    titles = [e.find_element_by_class_name("txt").text for e in elems]

    sale_flag = df["title"].isin(titles)
    if not any(sale_flag):
        df = get_sample(driver, elems, df)
        sale_flag = df["title"].isin(titles)
    print(df)

    use_min = np.min(df["use"].values)
    #使用回数が最小かつ動画があるもの
    choice_idx = np.argmax(1. * (df["use"] == use_min) * sale_flag * ~df["video_path"].isnull() * np.random.rand(len(df)))

    url = df.loc[choice_idx, "page_url"]
    driver.get(url)
    sleep(1)

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
    
    
    url = get_ad_twitter_link(driver)
    
    df.loc[choice_idx, "ad_url"] = url
    df.loc[choice_idx, "special"] = "@@@".join(special_text)
    driver.quit()
    return choice_idx

def clean_tag(hashtag):
    hashtag = re.sub(r'\([^()]*\)', '', hashtag)
    hashtag = hashtag.replace("#▼すべて表示する", "")
    return hashtag



args = get_args()
csv_path = f"C:\\Users\\hotta_mini\\affi_data\\data_4111.csv"

if  os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, encoding="shift-jis")
else:
    df = None
    print("データがありません")
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