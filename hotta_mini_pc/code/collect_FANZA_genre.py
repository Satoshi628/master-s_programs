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

from utils import GENRE_DICT


CLICK_TIME = 0.1

def get_args():
    parser = argparse.ArgumentParser(description="FANZAデータ収集プログラム(ジャンル別)")
    parser.add_argument(
        "-g", "--genre", default="", help="Genre")
    args = parser.parse_args()
    
    if not args.genre in GENRE_DICT:
        raise ValueError(f"Genreは{list(GENRE_DICT.keys())}から選択してください")
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


def get_sample(driver, df, csv_path):

    #"このリストを保存する"が一番最初にくるため削除
    product_elems = driver.find_elements_by_class_name("txt")[1:]

    dataframe_list = []

    for prod_e in product_elems:
        data = dict()
        #作品一つクリック
        prod_e.click()

        #クーポン表示削除
        xpath_click(driver, "//*[@id=\"campaign-popup-close\"]/img")

        #サンプル動画表示クリック
        sample_flag = xpath_click(driver, "//*[@id=\"detail-sample-movie\"]/div/a")

        #情報取得
        info = get_info(driver)
        if df is not None:
            if (df["title"] == info["title"]).any():
                df.loc[df["title"] == info["title"], "special"] = info["special"]
                driver.back()
                sleep(1)
                continue
        
        data.update(info)
        data.update(get_review(driver))

        #ダウンロード
        if sample_flag:
            video_flag, video_url, save_path  = video_download()
            data["video_url"] = video_url
            if video_flag:
                data["video_path"] = save_path
            pass
        else:
            data["video_url"] = None
            data["video_path"] = None

        #広告リンク作成
        url = get_ad_twitter_link(driver)
        data["ad_url"] = url
        data["use"] = 0
        dataframe_list.append(data)
        driver.back()
        sleep(1)

        #保存
        new_df = pd.DataFrame(dataframe_list)

        if df is not None:
            df = pd.concat([df, new_df])
        else:
            df = new_df

        #titleの重複対策
        df = df[~df.duplicated(subset='title')]
        
        try:
            df.to_csv(csv_path, encoding="shift-jis", index=False)
        except Exception as e:
            print(f"ERROR:save csv")
            print(e)
            df.to_csv(csv_path, encoding="shift-jis", errors='ignore', index=False)


args = get_args()
csv_path = f"C:\\Users\\hotta_mini\\affi_data\\data_{GENRE_DICT[args.genre]}.csv"

if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, encoding="shift-jis")
else:
    df = None

options = Options()
# linux上で動かすときは次の2つも実行してください。
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')

driver = webdriver.Chrome(options=options)

#FANZA動画ランキングに移動
url = f"https://www.dmm.co.jp/digital/videoa/-/list/=/article=keyword/id={GENRE_DICT[args.genre]}" + "/page={}/"

driver.get(url.format(1))

#年齢確認
xpath_click(driver, "//*[@id=\"dm-content\"]/main/div/div/div[2]")

#ログイン
FANZA_login()

#アフィバー表示
class_click(driver, "c-toolbar__open")
get_sample(driver, df, csv_path)

if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, encoding="shift-jis")
else:
    df = None

driver.quit()

# "//*[@id=\"main-ds\"]/div[2]/table[1]/tbody/tr/td[1]/div/p/a"
# "//*[@id=\"main-ds\"]/div[2]/table[1]/tbody/tr/td[1]/a"
# "//*[@id=\"main-ds\"]/div[2]/table[2]/tbody/tr/td[1]/a"
# "//*[@id=\"main-ds\"]/div[2]/table[1]/tbody/tr/td[2]/a"
