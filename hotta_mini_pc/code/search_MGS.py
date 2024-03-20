import os
import sys
import re
import datetime
import argparse
import requests
import urllib.parse

from glob import glob
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
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
    parser.add_argument("-s", "--savepath", default="C:\\Users\\hotta_mini\\affi_data\\data_MGS.csv", help="save csv path")
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

class MGS_Data_Collector():
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
        self.driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))
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
    
    def MGS_login(self):
        self.driver.get("https://www.mgstage.com/login/login.php")

        #年齢確認
        self.confirm_age()

        self.wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id=\"login-btn\"]")))
        self.driver.find_element(By.XPATH, "//*[@id=\"inputEmail\"]").send_keys("a.affiliate0830@gmail.com")
        self.driver.find_element(By.XPATH, "//*[@id=\"inputPassword\"]").send_keys("meicraft")
        sleep(1)
        xpath_click(self.driver, "//*[@id=\"login-btn\"]")
        self.wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id=\"search_submit\"]")))

    def search(self):
        serch_query = self.args.query + " -セット商品 -福袋 -4時間以上作品 -16間以上作品" if self.args.no_long else self.args.query
        serch_query = urllib.parse.quote(serch_query)
        url = f"https://www.mgstage.com/search/cSearch.php?search_word={serch_query}&type=top&page=1"

        self.driver.get(url)
        sleep(1)
    
    def confirm_age(self):
        #年齢確認
        xpath_click(self.driver, "//*[@id=\"AC\"]")
        sleep(1)
    
    def get_product_elems(self):
        elems = self.driver.find_elements(By.XPATH, "//*[@id=\"center_column\"]/div[2]/div/ul/*")
        return elems
    
    def get_product(self):
        #サンプル動画取得
        video_dict = self.download_video()

        #値段取得
        price_dict = self.get_price()

        #レビュー取得
        review_dict = self.get_review()

        #情報取得
        info_dict = self.get_info()

        #サンプル画像取得
        image_paths = self.get_sample_images(info_dict["product_number"])

        #アフィリエイトリンクを取得
        affi_dict = self.get_affiliate_link()

        #現在時刻、使用回数を取得
        meta_dict = {"get_time":datetime.datetime.now(), "use": 0}


        #None: Python>=3.5
        data = {**info_dict, **price_dict, **meta_dict, **image_paths, **video_dict, **affi_dict, **review_dict}

        return data

    def get_affiliate_link(self):
        click_flag = xpath_click(self.driver, "//*[@id=\"center_column\"]/div[1]/div[1]/div/div/a")
        if not click_flag:
            xpath_click(self.driver, "//*[@id=\"playing\"]/p/a[2]")

        sleep(0.5)
        url = self.driver.find_element(By.XPATH, "//*[@id=\"link_copy\"]").get_attribute("data-clipboard-text")
        #linktool => twitterlink
        url = url.replace("linktool", "twitterlink")
        sleep(0.5)
        return {"ad_url": url}
    
    def get_price(self):
        # "//*[@id=\"streaming_price\"]"
        "//*[@id=\"PriceList\"]/label[1]"
        elem = self.driver.find_elements(By.XPATH, "//*[@id=\"PriceList\"]/label[*]")[-1]

        price = elem.find_elements(By.XPATH, "//*[@id=\"streaming_price\"]")[-1].text
        price = int(price.replace(",", "").replace("円(税込)", ""))
        return {"price": price}

    def get_info(self):
        #お気に入り登録
        if len(self.driver.find_elements(By.XPATH, "//*[@id=\"center_column\"]/div[1]/div[1]/div/table[2]")) != 0:
            bookmark_num = self.driver.find_element(By.XPATH, "//*[@id=\"center_column\"]/div[1]/div[1]/div/table[1]/tbody/tr[1]/td[2]").text.replace(",", "")
            bookmark_num = int(re.sub(r"\D", "", bookmark_num))
            elem = self.driver.find_element(By.XPATH, "//*[@id=\"center_column\"]/div[1]/div[1]/div/table[2]")
        else:
            bookmark_num = self.driver.find_element(By.XPATH, "//*[@id=\"playing\"]/dl[1]").text.replace(",", "")
            bookmark_num = int(re.sub(r"\D", "", bookmark_num))
            elem = self.driver.find_element(By.XPATH, "//*[@id=\"center_column\"]/div[1]/div[1]/div/table")
        
        infomations = [e.text for e in elem.find_elements(By.XPATH, "tbody/tr[*]")]

        #現在のURLを保存
        current_URL = self.driver.current_url
        
        #タイトル
        title = text_convert(self.driver.find_element(By.XPATH, "//*[@id=\"center_column\"]/div[1]/h1").text)

        #セール情報取得
        sale_text = "セール中" if len(self.driver.find_elements(By.CLASS_NAME, "btn_sp_big")) != 0 else ""

        #出演者
        actress = "".join([info for info in infomations if "出演： " in info]).replace("出演： ", "")

        #ジャンル
        Genres = "".join([info for info in infomations if "ジャンル： " in info]).replace("ジャンル： ", "").replace(" ", "@@@")

        #収録時間
        video_time = "".join([info for info in infomations if "収録時間： " in info]).replace("収録時間： ", "").replace("min", "")

        #メーカー
        maker = "".join([info for info in infomations if "メーカー： " in info]).replace("メーカー： ", "")
        
        #品番
        prodct_num = "".join([info for info in infomations if "品番： " in info]).replace("品番： ", "")

        return {"title": title, "product_number": prodct_num,"page_url": current_URL, "actress": actress, "maker": maker,
                "video_time": video_time, "genre": Genres, "bookmark_number": bookmark_num,"sale": sale_text}

    def get_review(self):
        
        #レビュー表示ボタンを押す
        xpath_click(self.driver, "review_display")
        
        reviews = []
        elems = self.driver.find_elements(By.XPATH, "//*[@id=\"user_review\"]/ul/li[*]")
        for e in elems:
            if len(e.find_elements(By.CLASS_NAME, "star_50")) > 0 or len(e.find_elements(By.CLASS_NAME, "star_40")) > 0:
                text = e.find_element(By.CLASS_NAME, "text").text
                reviews.append(text_convert(text))

        return {"reviews": "@@@".join(reviews)}

    def get_sample_images(self, prodct_num):
        xpath_click(self.driver, "//*[@id=\"center_column\"]/div[1]/div[1]/div/div/h2/img")
        sleep(1.75)

        image_paths = []
        save_paths = []
        save_dir = f"C:\\Users\\hotta_mini\\affi_data\\images_MGS\\{prodct_num}"
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        else:
            xpath_click(self.driver, "//*[@id=\"lightbox-secNav-btnClose\"]")
            save_paths = glob(os.path.join(save_dir, "*"))
            return {"sample_images_url": "", "sample_images_path": "@@@".join(save_paths)}
        
        n = 0
        while True:
            img_url = self.driver.find_element(By.XPATH, "//*[@id=\"lightbox-image-copy\"]").get_attribute("src")
            image_paths.append(img_url)
            r = requests.get(img_url, stream=True, timeout=10)
            r.raise_for_status()
            ope = os.path.splitext(img_url)[-1]
            save_paths.append(os.path.join(save_dir, f"{n:04}{ope}"))
            with open(os.path.join(save_dir, f"{n:04}{ope}"), 'wb') as f:
                f.write(r.content)
            
            next_flag = xpath_click(self.driver, "//*[@id=\"lightbox-nav-btnNext-text\"]")
            if next_flag:
                n += 1
                sleep(1.5)
            else:
                xpath_click(self.driver, "//*[@id=\"lightbox-secNav-btnClose\"]")
                break
        
        return {"sample_images_url": "@@@".join(image_paths), "sample_images_path": "@@@".join(save_paths)}
    
    def download_video(self):
        sample_flag = class_click(self.driver, "button_sample")

        if not sample_flag:
            xpath_click(self.driver, "//*[@id=\"sampleLayer\"]/div[1]/a/img")
            print(self.driver.current_url)
            return {"video_url": None, "video_path": None}
        sleep(1.5)
        ## 動画のiframeに移動

        video_elem = self.driver.find_element(By.XPATH, "//*[@id=\"SamplePlayerPane\"]/div/video/source")
        video_url = video_elem.get_attribute("src")

        file_name = video_url.split("/")[-1].split(".")[0]
        save_path = f"C:\\Users\\hotta_mini\\affi_data\\data_MGS\\{file_name}.mp4"
        
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
        
        xpath_click(self.driver, "//*[@id=\"sampleLayer\"]/div[1]/a/img")
        sleep(0.5)

        if result:
            
            return {"video_url": video_url, "video_path": save_path}
        else:
            return {"video_url": video_url, "video_path": None}


    def __call__(self):
        print("csv open")
        self.open_csv()

        print("driver open")
        self.open_driver()

        print("MGS login")
        self.MGS_login()

        print(f"search {self.args.query}")
        self.search()

        print("get product elems")
        elems = self.get_product_elems()
        product_links = [e.find_element(By.TAG_NAME, "a").get_attribute("href") for e in elems]


        #作品一つクリック
        for link in tqdm(product_links, leave=False):
            self.driver.get(link)
            sleep(1)
            print("get product")
            data = self.get_product()
            
            self.df = pd.concat([self.df, pd.DataFrame([data])])

            #品番の重複対策
            self.df = self.df[~self.df.duplicated(subset='product_number')]
            for key, value in data.items():
                if value:
                    self.df.loc[self.df["product_number"] == data["product_number"], key] = value

            self.save_csv()

        print("driver close")
        self.close_driver()


    def __del__(self):
        if self.driver is not None:
            self.driver.quit()


if __name__ == "__main__":
    args = get_args()
    MDC = MGS_Data_Collector(args)
    MDC()