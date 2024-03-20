import os
import sys
import re
import datetime
import argparse
import requests
import urllib.parse

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
import chromedriver_binary
from time import sleep
import pandas as pd

from edit_video import crop_video, delete_video
from utils import GENRE_DICT, AFI_ID_DICT
from KONOYONOOWARI import KONOYONOOWARI_HENSU

CLICK_TIME = 0.1
X_moji_LIMIT = 140

def get_args():
    parser = argparse.ArgumentParser(description="FANZAからセール品を収集(検索)")
    parser.add_argument("-q", "--query", default="", help="search")
    parser.add_argument("-s", "--savepath", default="C:\\Users\\hotta_mini\\affi_data\\data.csv", help="save csv path")
    parser.add_argument("--no_long", action='store_true')
    args = parser.parse_args()
    
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

class FANZA_Data_Collector():
    def __init__(self, args):
        self.args = args
        self.driver = None
    
    def open_driver(self):
        options = Options()
        # linux上で動かすときは次の2つも実行してください。
        options.add_argument('--headless')
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
    
    def FANZA_login(self):
        self.driver.get("https://accounts.dmm.co.jp/service/login/password")
        sleep(2)
        self.driver.find_element(By.XPATH, "//*[@id=\"login_id\"]").send_keys("a.affiliate0830@gmail.com")
        self.driver.find_element(By.XPATH, "//*[@id=\"password\"]").send_keys("meicraft")
        sleep(1)
        xpath_click(self.driver, "//*[@id=\"loginbutton_script_on\"]/span/input")
        sleep(1)

    def search(self):
        serch_query = self.args.query + " -セット商品 -福袋 -4時間以上作品 -16間以上作品" if self.args.no_long else self.args.query
        serch_query = urllib.parse.quote(serch_query)

        url = f"https://www.dmm.co.jp/digital/videoa/-/list/search/=/?searchstr={serch_query}"

        self.driver.get(url)
        sleep(1)
    
    def confirm_age(self):
        #年齢確認
        xpath_click(self.driver, "//*[@id=\"dm-content\"]/main/div/div/div[2]")
        sleep(1)
    
    def del_coupon(self):
        coupon_xpath = "//*[@id=\"campaign-popup-close\"]/img"
        if len(self.driver.find_elements(By.XPATH, coupon_xpath)):
            #クーポン表示削除
            xpath_click(self.driver, coupon_xpath)
            sleep(1)

    def get_product_elems(self):
        elems = self.driver.find_elements(By.XPATH, "//*[@id=\"list\"]/li[*]")
        return elems
    
    def get_product(self):
        #クーポン表示を削除
        self.del_coupon()
        sleep(0.25)

        
        #サンプル動画取得
        video_dict = self.download_video()

        #値段取得
        price_dict = self.get_price()

        #情報取得
        info_dict = self.get_info()

        #レビュー取得
        review_dict = self.get_review()

        #アフィリエイトリンクを取得
        affi_dict = self.get_affiliate_link()

        #現在時刻、使用回数を取得
        meta_dict = {"get_time":datetime.datetime.now(), "use": 0}

        #None: Python>=3.5
        data = {**info_dict, **price_dict, **meta_dict, **video_dict, **affi_dict, **review_dict}

        return data

    def get_affiliate_link(self):
        xpath_click(self.driver, "//*[@id=\"affiliate-toolbar\"]/div[1]/ul/li[1]")

        sleep(0.5)
        html = self.driver.find_element(By.CLASS_NAME, "c-contents__source").text
        url = re.findall('(?<=href=\").+?(?=\")', html)[0].replace("package_text_large", "twitter")
        sleep(0.25)
        class_click(self.driver, "c-icoClose")
        sleep(0.5)
        return {"ad_url": url}
    
    def get_price(self):
        elem = self.driver.find_elements(By.XPATH, "//*[@id=\"basket_contents\"]/div[1]/form/ul/li[*]")[-1]

        if len(elem.find_elements(By.CLASS_NAME, "tx-hangaku")):
            price = elem.find_element(By.CLASS_NAME, "tx-hangaku").text
        else:
            price = elem.find_element(By.CLASS_NAME, "price").text
        
        price = int(price.replace(",", "").replace("円", ""))
        return {"price": price}

    def get_info(self):
        infomations = [e.text for e in self.driver.find_elements(By.XPATH,  "//*[@id=\"mu\"]/div/table/tbody/tr/td[1]/table/tbody/tr[*]")]

        #現在のURLを保存
        current_URL = self.driver.current_url
        
        #出演者
        actress = "".join([info for info in infomations if "出演： " in info or "名前： " in info]).replace("出演： ", "").replace("名前： ", "")

        #ジャンル
        Genres = "".join([info for info in infomations if "ジャンル： " in info]).replace("ジャンル： ", "").replace(" ", "@@@")

        #収録時間
        video_time = "".join([info for info in infomations if "収録時間： " in info]).replace("収録時間： ", "").replace("min", "")

        #メーカー
        maker = "".join([info for info in infomations if "メーカー： " in info or "レーベル： " in info]).replace("メーカー： ", "").replace("レーベル： ", "")
        
        #品番
        prodct_num = "".join([info for info in infomations if "品番： " in info]).replace("品番： ", "")


        elem = self.driver.find_element(By.CLASS_NAME, "hreview")

        try:
            title = text_convert(elem.find_element(By.ID,  "title").text)
        except:
            print(f"ERROR:title text")
            title = None

        #セール情報取得
        try:
            sale_text = text_convert(elem.find_element(By.CLASS_NAME, "tx-hangaku").text)
        except:
            sale_text = ""
            print(f"ERROR:hangaku text")
        
        #お気に入り登録
        bookmark_num = int(self.driver.find_element(By.XPATH, "//*[@id=\"mu\"]/div/table/tbody/tr/td[1]/div[2]/p/span/span").text)

        return {"title": title, "product_number": prodct_num,"page_url": current_URL, "actress": actress, "maker": maker,
                "video_time": video_time, "genre": Genres, "bookmark_number": bookmark_num,"sale": sale_text}

    def get_review(self):
        #評価が高い順を選択
        try:
            elem = self.driver.find_element(By.XPATH, '//*[@id=\"review\"]/div[2]/div/div[3]/div[1]/div[1]/div/select')
        except:
            return {"reviews": ""}
        select = Select(elem)
        select.select_by_index(2)
        sleep(1)
        
        #レビュー表示ボタンをすべて押す
        elems = self.driver.find_elements(By.CLASS_NAME, "d-modtogglelink-open")
        for e in elems:
            e.click()
        
        #ページ最上にスクロール
        sleep(1)
        self.driver.execute_script('window.scrollTo(0, 0);')
        
        reviews = []
        elems = self.driver.find_elements(By.CLASS_NAME, "d-review__unit")
        for e in elems:
            if len(e.find_elements(By.CLASS_NAME, "d-rating-50")) > 0 or len(e.find_elements(By.CLASS_NAME, "d-rating-40")) > 0:
                text = e.find_element(By.CLASS_NAME, "d-review__unit__comment").text
                reviews.append(text_convert(text))
        # elems = self.driver.find_elements(By.CLASS_NAME, "d-review__unit__comment")
        # reviews = [text_convert(e.text) for e in elems]
        return {"reviews": "@@@".join(reviews)}

    def download_video(self):
        sample_flag = xpath_click(self.driver, "//*[@id=\"detail-sample-movie\"]/div/a")
        if not sample_flag:
            return {"video_url": None, "video_path": None}
        sleep(1.5)
        ## 動画のiframeに移動
        iframe = self.driver.find_element(By.XPATH, "//*[@id=\"DMMSample_player_now\"]")
        self.driver.switch_to.frame(iframe)

        video_elem = self.driver.find_element(By.TAG_NAME,  "video")
        video_url = video_elem.get_attribute("src")

        ### 元のフレームに戻る
        self.driver.switch_to.default_content()
        
        file_name = video_url.split("/")[-2]
        save_path = f"C:\\Users\\hotta_mini\\affi_data\\data\\{file_name}.mp4"
        
        result = False
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
                print(f'***** requests エラー({e}): {i + 1}/5')
                sleep(1)
            else:
                result = True
                break  # try成功
        if result:
            return {"video_url": video_url, "video_path": save_path}
        else:
            return {"video_url": video_url, "video_path": None}


    def __call__(self):
        print("csv open")
        self.open_csv()

        print("driver open")
        self.open_driver()

        print("Fanza login")
        self.FANZA_login()

        print("confirm age")
        self.confirm_age()

        print("affiliate bar")
        #アフィバー表示
        class_click(self.driver, "c-toolbar__open")
        sleep(0.5)

        print(f"search {self.args.query}")
        self.search()

        print("get product elems")
        elems = self.get_product_elems()
        product_links = [e.find_element(By.TAG_NAME,  "a").get_attribute("href") for e in elems]

        #作品一つクリック
        for link in tqdm(product_links, leave=False):
            self.driver.get(link)
            sleep(1)
            print("get product")
            data = self.get_product()
            
            self.df = pd.concat([self.df, pd.DataFrame([data])])

            #品番の重複対策
            self.df = self.df[~self.df.duplicated(subset='product_number')]

            self.save_csv()

        print("driver close")
        self.close_driver()


    def __del__(self):
        if self.driver is not None:
            self.driver.quit()


if __name__ == "__main__":
    args = get_args()
    FDC = FANZA_Data_Collector(args)
    FDC()