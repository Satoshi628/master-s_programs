# coding: UTF-8

import os
import json
import argparse
import requests
import urllib.parse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary
from time import sleep

meker_list = ["Aircontrol","BeFree","E-BODY","Fitch","HHH","kawaii*","kira☆kira","MVG","OPERA","OPPAI","ROOKIE","Ｖ","アタッカーズ","えむっ娘ラボ","ダスッ！","ナンパJAPAN","はじめ企画","ビビアン","ワンズファクトリー","痴女ヘヴン","変態紳士倶楽部","本中","未満","無垢","溜池ゴロー","ｼﾛｳﾄ逸材発掘～仕事帰りのﾔﾘもくちゃんSSR","職業女子","Buzzシロウト","GO！GO！お手当ちゃん","MOON LIGHTING","ギャルpay","しろうとエチチ.ch","しろうとヤッホー","しろうと屋","チョロすぎQ","ニュージェネ","ねっとりフリックス","巨乳は飲み物。","新世代女子","神級ビッチ","素人ごっそ","素人ぱいぱい","素人天然水","素人盗撮倶楽部","東京恋マチ女子","東狂ハメンジャーズ","肉女子キュンキュン♪","日払いちゃん","素人ムクムク","＃シロウト逸材発掘","素人ぷるるん","職業女子","素人ムクムク 夢中","路地裏ぱんぱん","シロウト速報","素人ムクムク-塩-","素人盗撮俱楽部","素人ChuChu","素人ムクムク-夢中-","裏垢ドットえす","アイデアポケット","PREMIUM","Madonna","MOODYZ","グローバルメディアエンタテインメント","グローバルメディアアネックス","桃太郎映像出版","バビロン/妄想族","ニイハオサイコウ/妄想族","全日本カメコ協同組合/妄想族","KSB企画/エマニエル","イノセント/妄想族","ドグマ","こぐま/妄想族","綜実社/妄想族","コレ彦/妄想族","コレ彦","かつお物産/妄想族","ゲインコーポレーション","堅者の食卓/妄想族","AVS collector’s","美人魔女/エマニエル","アイドリ/妄想族","稀（まれ）/妄想族","BabyEntertainment","アクアモール/エマニエル","有閑ミセス/エマニエル","熟女塾/エマニエル","スパルタン/妄想族","ABC/妄想族","TEPPAN","縦動画プロジェクト","ぽかぽか/妄想族","HYBRID映像/妄想族","かぐや姫Pt/妄想族","ミセスの素顔/エマニエル","LOVEま○こ/妄想族","まんげつ/妄想族","Vrevo","姦乱者/妄想族","バルタン","SEX Agent/妄想族","ZETTON","Pandora/エマニエル","ILLEGAL＜イリーガル＞/妄想族","人妻援護会/エマニエル","VENUS","熟女JAPAN","ハメドリネットワークSecondEdition","パーフェクトコミュニケーションズ","BRAVO","平日14時の発情妻たち","ヒプノシスラボ/妄想族","熟道","h.m.p DORAMA","アロマ企画","大塚フロッピー","ケチャラパチャラ/妄想族","マックスエー","HyakkinTV","MAX-Aレジェンド","ION/妄想族","アリスJAPAN","Smartmedia production/妄想族","PETSHOP/妄想族","ミル","private mask/妄想族","Asia/妄想族","DIVA/妄想族","P-BOX VR","毒宴会","4K VR","MONDELDE VR","肉盛","Cosmo Planets VR","クレイジーウォーカー","VR buz","キネマ座","通勤快速","フェラすぺ","生ハメ素人ch","こすパコハメ撮りおじさん","KMPVR-彩-","犬/妄想族","うさぎ/妄想族","ビッグ・ザ・肉道/妄想族","イルカ/エマニエル","グローリークエスト","グローリークエストVR","ヒプノシスRASH","煩悩組/妄想族","催●RED","人妻文化センター/エマニエル","ディープス","激レア素人ちゃん","STUDIO I’s/妄想族","AMATEUR BOX/妄想族","軟派舎/妄想族","BRAVO/ミスターインパクト","unfinished","MARRION","しろうとがーる/妄想族","S-Cute","下半身タイガース/妄想族","レアルワークス","僕たち男の娘","宇宙企画","ケイ・エム・プロデュース","BAZOOKA","メディアステーション","パコパコ団とゆかいな仲間たち/妄想族","山と空/妄想族","KMPVR-bibi-","スクープ","S級素人","V＆R PRODUCE","なでしこ","エロタイム","Z-MEN","NAGIRA","ヒメゴト","RADICAL-KMPVR-","GIGOLO（ジゴロ）","椿鳳院","ステルス","ナンパHEAVEN","むちゃぶりTV","サロメ","地雷系女子","ルーナ旬香舎","令和四天王","東京恋人","スリーサウザンド","忍","タイガーマイスターズ","新世紀文藝社","マダムス","トップマーシャル","REAL VR-Neo-","300 Three Hundred","カメラ小僧","エロガチャ","絆書房","初めてのAV出演","これすこ。","バリカワ","B級熟女選手権","HEAVEN","日本藝術浪漫文庫","刺激ストロング","港区女子","THE BEST OF 3DVR","日本近代ロマン書房","世田谷VR","＆RiBbON","アダム書房","ボリューミー","S級素人VR -DX-","裸王","横浜かまちょ","スカッド","ナタリー文庫","月刊盗撮現代","VRスタジアム","俺の素人","ゲリラ","ゾクゾク娘/妄想族","レインボー/妄想族","GALDQN/妄想族","B-hole/エマニエル","EROTICA","アップス","カルマ","熟女はつらいよ/熟女卍","熟女大学/熟女卍","素人まっちんぐEX/妄想族","ゑびすさん/妄想族","ゆりえっち/妄想族","熟の蔵/エマニエル","FAプロ","ジェントルマン/妄想族","新セカイ/妄想族","MUTEKI","S1","濡壺/妄想族","キチックス/妄想族","乳と母/エマニエル","Lady Boy/妄想族","肉厚食堂/妄想族","幼獄LiTE/妄想族","ちぃぱいペチャ子/妄想族","カウカウパラダイス/妄想族","おっぱいデカ美/妄想族","CREAM SODA/妄想族","宝石箱/妄想族","苺一会/妄想族","ルネピクチャーズ","豊彦"]


def xpath_click(driver, xpath):
    try:
        elem = driver.find_element_by_xpath(xpath)
        elem.click()
        sleep(CLICK_TIME)
    except:
        print(f"ERROR:xpath_click {xpath}")
        return False
    return True

options = Options()
# linux上で動かすときは次の2つも実行してください。
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')

driver = webdriver.Chrome(options=options)
driver.get("https://www.dmm.co.jp/digital/videoa/-/list/search/=/?searchstr=Aircontrol%20-VR")

xpath_click(driver, "//*[@id=\"dm-content\"]/main/div/div/div[2]")
sleep(1)

maker_dict = {}
for meker in meker_list:
    serch_query = meker + " -VR -セット商品 -福袋 -4時間以上作品 -16間以上作品"
    serch_query = urllib.parse.quote(serch_query)

    url = f"https://www.dmm.co.jp/digital/videoa/-/list/search/=/?searchstr={serch_query}"

    driver.get(url)
    sleep(2)
    text = driver.find_element_by_class_name("list-boxcaptside").text.replace(",", "")
    try:
        maker_dict[meker] = int(text.split("タイトル中")[0])
    except:
        maker_dict[meker] = 0

    with open("meker.json", mode="w", encoding='UTF-8') as f:
        json.dump(maker_dict, f, indent=2, ensure_ascii=False)
