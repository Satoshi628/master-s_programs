import requests
from bs4 import BeautifulSoup
import numpy as np
from time import sleep
import pandas as pd
import pyodbc
import re

#完成！
def update_T_Dataset():
    Dataset_name_dict =[]
    url = "https://paperswithcode.com/datasets?page={}"
    html = requests.get(url.format("1"))
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")
    #論文パラメータ作成
    results_count = int(re.sub(r"\D", "", soup.find("h1",class_="results-count").text))
    Dataset_name = [item.text.replace("\n","").replace("  ","").replace("'","''") for item in soup.find_all("span",class_="name")]
    Dataset_name_dict.extend(Dataset_name)
    print("1 ページ完了")
    data_per_page =len(Dataset_name)

    for i in range(2,results_count//data_per_page+2):
        sleep(1)
        print(i,"ページ完了")
        html = requests.get(url.format(i))
        #print(html.url)
        assert html.status_code == 200,"アクセスに失敗しました"
        soup = BeautifulSoup(html.text, features="html.parser")
        #論文パラメータ作成
        Dataset_name = [item.text.replace("\n","").replace("  ","").replace("'","''") for item in soup.find_all("span",class_="name")]
        Dataset_name_dict.extend(Dataset_name)
    print(len(Dataset_name_dict))
    Dataset_dict = [{"データセット名":Dataset_name_dict[i]} for i in range(len(Dataset_name_dict))]
    df_new = pd.DataFrame(Dataset_dict)
    #自信のデータの重複削除
    df_new =df_new.drop_duplicates(subset="データセット名")
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=C:\Users\hotta_mini\Desktop\Python\database\論文.accdb;'
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    #T_DatasetからDataFrameを取得
    sql = 'SELECT * FROM T_Dataset'
    df_old = pd.read_sql(sql, conn)
    df_old['データセット名'] =df_old['データセット名'].str.replace("'","''")

    #データセット名で重複してはいけないため2つのDataの重複削除
    df_new = df_new[~df_new['データセット名'].isin(df_old['データセット名'])]

    sql = "INSERT INTO T_Dataset(データセット名) VALUES('{}')"
    for i in range(len(df_new)):
        cursor.execute(sql.format(df_new.iat[i,0]))
        cursor.commit()

    print("{}コのデータを更新しました".format(len(df_new)))

    cursor.close()
    conn.close()

#完成！
#入力の形、'は''に変換しない
#空白のばあいNullと記載
#paper_new = {発行年:number,Meeting:str,論文名:str,URL:str}
#PtA_new = {論文ID:str,著者ID:str}
#PtD_new = {論文ID:str,データセットID:str}
def update_table(paper_new,PtA_new,PtD_new=None):
    paper_new['論文名'] = paper_new['論文名'].str.replace("'","''")
    paper_new['Meeting'] = paper_new['Meeting'].str.replace("'","''")
    PtA_new['論文ID'] = PtA_new['論文ID'].str.replace("'","''")
    PtA_new['著者ID'] = PtA_new['著者ID'].str.replace("'","''")
    if PtD_new != None:
        PtD_new['論文ID'] = PtD_new['論文ID'].str.replace("'","''")
        PtD_new['データセットID'] = PtD_new['データセットID'].str.replace("'","''")

    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=C:\Users\hotta_mini\Desktop\Python\database\論文.accdb;'
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    #-------------------- テーブルの値取得 --------------------#
    #T_論文読み込み
    sql = 'SELECT * FROM T_論文'
    paper_old = pd.read_sql(sql, conn)
    paper_old['論文名'] =paper_old['論文名'].str.replace("'","''")
    #T_Meeting読み込み
    sql = 'SELECT * FROM T_Meeting'
    Meeting_old = pd.read_sql(sql, conn)
    Meeting_old['Meeting名'] =Meeting_old['Meeting名'].str.replace("'","''")
    #T_論文to入力読み込み
    sql = 'SELECT * FROM T_論文to入力'
    PtD_old = pd.read_sql(sql, conn)
    #T_論文to著者読み込み
    sql = 'SELECT * FROM T_論文to著者'
    PtA_old = pd.read_sql(sql, conn)
    #T_Dataset読み込み
    sql = 'SELECT * FROM T_Dataset'
    Dataset_old = pd.read_sql(sql, conn)
    Dataset_old['データセット名'] =Dataset_old['データセット名'].str.replace("'","''")
    #T_著者読み込み
    sql = 'SELECT * FROM T_著者'
    author_old = pd.read_sql(sql, conn)
    author_old['名前'] =author_old['名前'].str.replace("'","''")

    #.................... テーブルの値取得 ....................#

    #-------------------- Meetingテーブル更新 --------------------#
    #T_Meeting(Meeting名)は重複してはいけいない
    Meeting_new = paper_new['Meeting'][~paper_new['Meeting'].str.lower().isin(Meeting_old['Meeting名'].str.lower())]
    Meeting_new = Meeting_new.drop_duplicates()

    #NULL(nan)をなくす
    Meeting_new = Meeting_new[~Meeting_new.isnull()]
    Meeting_new = Meeting_new[~(Meeting_new == 'Null')]

    #重複していないMeetingをT_Meetingに書き込む
    sql = "INSERT INTO T_Meeting(Meeting名) VALUES('{}')"
    for item in Meeting_new:
        cursor.execute(sql.format(item))
        cursor.commit()
    print("{}コ、T_Meetingを更新しました".format(len(Meeting_new)))

    #T_Meeting情報更新
    sql = 'SELECT * FROM T_Meeting'
    Meeting_old = pd.read_sql(sql, conn)
    Meeting_old['Meeting名'] =Meeting_old['Meeting名'].str.replace("'","''")

    #.................... Meetingテーブル更新 ....................#

    #-------------------- 論文テーブル更新 --------------------#
    #Meeting名をMeeting_IDに書き換える
    paper_new['Meeting'] = paper_new['Meeting'].str.lower().replace(Meeting_old['Meeting名'].str.lower().values.tolist(),Meeting_old['Meeting_ID'].values.tolist())
    #T_論文(論文名,URL)は重複してはいけいない
    paper_new = paper_new[~paper_new['論文名'].str.lower().isin(paper_old['論文名'].str.lower())]
    paper_new = paper_new.drop_duplicates(subset='論文名')
    paper_new = paper_new.drop_duplicates(subset='URL')

    #NULLとして書き込めるようにする
    paper_new =paper_new.replace(np.nan,'Null')

    sql = "INSERT INTO T_論文(発行年,Meeting,論文名,URL) VALUES({},{},'{}','{}')"
    for year,Meeting, paper,URL in zip(paper_new['発行年'],paper_new['Meeting'],paper_new['論文名'],paper_new['URL']):
        cursor.execute(sql.format(year,
                                Meeting,
                                paper,
                                URL))
        cursor.commit()
    print("{}コ、T_論文を更新しました".format(len(paper_new)))

    #T_論文情報更新
    sql = 'SELECT * FROM T_論文'
    paper_old = pd.read_sql(sql, conn)
    paper_old['論文名'] =paper_old['論文名'].str.replace("'","''")
    #.................... 論文テーブル更新 ....................#

    #-------------------- Datasetテーブル更新 --------------------#
    if PtD_new != None:
        #T_Dataset(データセット名)は重複してはいけいない
        Dataset_new = PtD_new['データセットID'][~PtD_new['データセットID'].isin(Dataset_old['データセット名'])]
        Dataset_new = Dataset_new.drop_duplicates()

        #NULL(nan)をなくす
        Dataset_new = Dataset_new[~Dataset_new.isnull()]

        #重複していないDatasetをT_Datasetに書き込む
        sql = "INSERT INTO T_Dataset(データセット名) VALUES('{}')"
        for item in Dataset_new:
            cursor.execute(sql.format(item))
            cursor.commit()
        print("{}コ、T_Datasetを更新しました".format(len(Dataset_new)))

        #T_Dataset情報更新
        sql = 'SELECT * FROM T_Dataset'
        Dataset_old = pd.read_sql(sql, conn)
        Dataset_old['データセット名'] =Dataset_old['データセット名'].str.replace("'","''")
    #.................... Datasetテーブル更新 ....................#

    #-------------------- 論文to入力テーブル更新 --------------------#
    if PtD_new != None:
        #論文名を論文IDに書き換える
        PtD_new['論文ID'] = PtD_new['論文ID'].str.lower().replace(paper_old['論文名'].str.lower().values.tolist(),paper_old['論文ID'].values.tolist())
        #データセット名をデータセットIDに書き換える
        PtD_new['データセットID'] = PtD_new['データセットID'].str.lower().replace(Dataset_old['データセット名'].str.lower().values.tolist(),Dataset_old['データセットID'].values.tolist())

        #元のデータと重複してはいけない
        PtD_new = PtD_new[~(PtD_new['論文ID'].isin(PtD_old['論文ID']) & PtD_new['データセットID'].isin(PtD_old['データセットID']))]
        PtD_new = PtD_new.drop_duplicates()

        #NULL(nan)をなくす
        PtD_new = PtD_new[~PtD_new.isnull()]

        #重複していないデータをT_論文to入力に書き込む
        sql = "INSERT INTO T_論文to入力(論文ID,データセットID) VALUES({},{})"
        for paper_id,dataset_id in zip(PtD_new['論文ID'],PtD_new['データセットID']):
            cursor.execute(sql.format(paper_id,dataset_id))
            cursor.commit()
        print("{}コ、T_論文to入力を更新しました".format(len(PtD_new)))
    #.................... 論文to入力テーブル更新 ....................#

    #-------------------- 著者テーブル更新 --------------------#
    #T_著者(名前)は重複してはいけいない
    author_new = PtA_new['著者ID'][~PtA_new['著者ID'].str.lower().isin(author_old['名前'].str.lower())]
    author_new = author_new.drop_duplicates()

    #NULL(nan)をなくす
    author_new = author_new[~author_new.isnull()]

    #重複していない名前をT_著者に書き込む
    sql = "INSERT INTO T_著者(名前) VALUES('{}')"
    for item in author_new:
        cursor.execute(sql.format(item))
        cursor.commit()
    print("{}コ、T_著者を更新しました".format(len(author_new)))

    #T_Dataset情報更新
    sql = 'SELECT * FROM T_著者'
    author_old = pd.read_sql(sql, conn)
    author_old['名前'] =author_old['名前'].str.replace("'","''")
    #.................... 著者テーブル更新 ....................#

    #-------------------- 論文to著者テーブル更新 --------------------#
    #論文名を論文IDに書き換える
    PtA_new['論文ID'] = PtA_new['論文ID'].str.lower().replace(paper_old['論文名'].str.lower().values.tolist(),paper_old['論文ID'].values.tolist())
    #名前を著者IDに書き換える
    PtA_new['著者ID'] = PtA_new['著者ID'].str.lower().replace(author_old['名前'].str.lower().values.tolist(),author_old['著者ID'].values.tolist())

    #元のデータと重複してはいけない
    PtA_new = PtA_new[~(PtA_new['論文ID'].isin(PtA_old['論文ID']) & PtA_new['著者ID'].isin(PtA_old['著者ID']))]

    #NULL(nan)をなくす
    PtA_new = PtA_new[~PtA_new.isnull()]

    #重複していないauthorをT_Datasetに書き込む
    sql = "INSERT INTO T_論文to著者(論文ID,著者ID) VALUES({},{})"
    for paper_id,author_id in zip(PtA_new['論文ID'],PtA_new['著者ID']):
        cursor.execute(sql.format(paper_id,author_id))
        cursor.commit()
    print("{}コ、T_論文to著者を更新しました".format(len(PtA_new)))
    #....................  論文to著者テーブル更新 ....................#
    cursor.close()
    conn.close()

def CVPR2021_scraping():
    url = "https://openaccess.thecvf.com/CVPR2021?day=2021-06-25"
    html = requests.get(url)
    #print(html.url)
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")

    #論文パラメータ作成
    title = [item.text for item in soup.find_all("dt",class_="ptitle")]
    pdf_URL = [item.get('href') for item in soup.find_all("a")]
    year = [int(re.findall(r'\d+',item.text)[0]) for item in soup.find_all("div",class_="bibref pre-white-space")]
    Meeting = ["CVPR"+str(item) for item in year]
    author = [item.text.replace("\n","").split(",") for item in soup.find_all("dd")][1::2]


    #最後は"back"と表記されている。
    del author[-1]

    #URL処理
    pdf_URL = [item for item in pdf_URL if isinstance(item,str)]
    pdf_URL = ["https://openaccess.thecvf.com"+item for item in pdf_URL if '.pdf' in item and 'papers' in item]


    assert len(title)==len(pdf_URL)==len(year)==len(Meeting)==len(author),"それぞれの要素の配列数が違います。"

    paper_dict = [{
        "発行年":year[i],
        "Meeting":'Null',
        "論文名":title[i],
        "URL":pdf_URL[i]
        }
        for i in range(len(title))]

    PtA_dict = [{
        "論文ID":title[i],
        "著者ID":author[i][j]
        }
        for i in range(len(title)) #これが先
        for j in range(len(author[i])) #こっちが前のforに内包されている
        ]

    return pd.DataFrame(paper_dict),pd.DataFrame(PtA_dict)

def CVPR2020_scraping():
    url = "https://openaccess.thecvf.com/CVPR2020?day=2020-06-18"
    html = requests.get(url)
    #print(html.url)
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")

    #論文パラメータ作成
    title = [item.text for item in soup.find_all("dt",class_="ptitle")]
    pdf_URL = [item.get('href') for item in soup.find_all("a")]
    year = [int(re.findall(r'\d+',item.text)[0]) for item in soup.find_all("div",class_="bibref")]
    author = [item.text.replace("\n","").split(",") for item in soup.find_all("dd")][1::2]

    #URL処理
    pdf_URL = [item for item in pdf_URL if isinstance(item,str)]
    pdf_URL = ["https://openaccess.thecvf.com/"+item for item in pdf_URL if '.pdf' in item and 'papers' in item]

    assert len(title)==len(pdf_URL)==len(year)==len(author),"それぞれの要素の配列数が違います。"

    paper_dict = [{
        "発行年":year[i],
        "Meeting":'Null',
        "論文名":title[i],
        "URL":pdf_URL[i]
        }
        for i in range(len(title))]

    PtA_dict = [{
        "論文ID":title[i],
        "著者ID":author[i][j]
        }
        for i in range(len(title)) #これが先
        for j in range(len(author[i])) #こっちが前のforに内包されている
        ]

    return pd.DataFrame(paper_dict),pd.DataFrame(PtA_dict)

def CVPR2019_scraping():
    url = "https://openaccess.thecvf.com/CVPR2019?day=2019-06-20"
    html = requests.get(url)
    #print(html.url)
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")

    #論文パラメータ作成
    title = [item.text for item in soup.find_all("dt",class_="ptitle")]
    pdf_URL = [item.get('href') for item in soup.find_all("a")]
    year = [int(re.findall(r'\d+',item.text)[0]) for item in soup.find_all("div",class_="bibref")]
    author = [item.text.replace("\n","").split(",") for item in soup.find_all("dd")][1::2]

    #URL処理
    pdf_URL = [item for item in pdf_URL if isinstance(item,str)]
    pdf_URL = ["https://openaccess.thecvf.com/"+item for item in pdf_URL if '.pdf' in item and 'papers' in item]

    assert len(title)==len(pdf_URL)==len(year)==len(author),"それぞれの要素の配列数が違います。"

    paper_dict = [{
        "発行年":year[i],
        "Meeting":'Null',
        "論文名":title[i],
        "URL":pdf_URL[i]
        }
        for i in range(len(title))]

    PtA_dict = [{
        "論文ID":title[i],
        "著者ID":author[i][j]
        }
        for i in range(len(title)) #これが先
        for j in range(len(author[i])) #こっちが前のforに内包されている
        ]

    return pd.DataFrame(paper_dict),pd.DataFrame(PtA_dict)

def CVPR2018_scraping():
    url = "https://openaccess.thecvf.com/CVPR2018?day=2018-06-19"
    html = requests.get(url)
    #print(html.url)
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")

    #論文パラメータ作成
    title = [item.text for item in soup.find_all("dt",class_="ptitle")]
    pdf_URL = [item.get('href') for item in soup.find_all("a")]
    year = [int(re.findall(r'\d+',item.text)[0]) for item in soup.find_all("div",class_="bibref")]
    author = [item.text.replace("\n","").split(",") for item in soup.find_all("dd")][1::2]

    #URL処理
    pdf_URL = [item for item in pdf_URL if isinstance(item,str)]
    pdf_URL = ["https://openaccess.thecvf.com/"+item for item in pdf_URL if '.pdf' in item and 'papers' in item]

    assert len(title)==len(pdf_URL)==len(year)==len(author),"それぞれの要素の配列数が違います。"

    paper_dict = [{
        "発行年":year[i],
        "Meeting":'Null',
        "論文名":title[i],
        "URL":pdf_URL[i]
        }
        for i in range(len(title))]

    PtA_dict = [{
        "論文ID":title[i],
        "著者ID":author[i][j]
        }
        for i in range(len(title)) #これが先
        for j in range(len(author[i])) #こっちが前のforに内包されている
        ]

    return pd.DataFrame(paper_dict),pd.DataFrame(PtA_dict)

def ICCV2019_scraping():
    url = "https://openaccess.thecvf.com/ICCV2019?day=2019-11-01"
    html = requests.get(url)
    #print(html.url)
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")

    #論文パラメータ作成
    title = [item.text for item in soup.find_all("dt",class_="ptitle")]
    pdf_URL = [item.get('href') for item in soup.find_all("a")]
    year = [int(re.findall(r'\d+',item.text)[0]) for item in soup.find_all("div",class_="bibref")]
    author = [item.text.replace("\n","").split(",") for item in soup.find_all("dd")][1::2]

    #URL処理
    pdf_URL = [item for item in pdf_URL if isinstance(item,str)]
    pdf_URL = ["https://openaccess.thecvf.com/"+item for item in pdf_URL if '.pdf' in item and 'papers' in item]

    assert len(title)==len(pdf_URL)==len(year)==len(author),"それぞれの要素の配列数が違います。"

    paper_dict = [{
        "発行年":year[i],
        "Meeting":'Null',
        "論文名":title[i],
        "URL":pdf_URL[i]
        }
        for i in range(len(title))]

    PtA_dict = [{
        "論文ID":title[i],
        "著者ID":author[i][j]
        }
        for i in range(len(title)) #これが先
        for j in range(len(author[i])) #こっちが前のforに内包されている
        ]

    return pd.DataFrame(paper_dict),pd.DataFrame(PtA_dict)

def ICCV2021_scraping():
    url = "https://openaccess.thecvf.com/ICCV2021?day=2021-10-13"
    html = requests.get(url)
    #print(html.url)
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")

    #論文パラメータ作成
    title = [item.text for item in soup.find_all("dt",class_="ptitle")]
    pdf_URL = [item.get('href') for item in soup.find_all("a")]
    year = [int(re.findall(r'\d+',item.text)[0]) for item in soup.find_all("div",class_="bibref pre-white-space")]
    Meeting = ["ICCV"+str(item) for item in year]
    author = [item.text.replace("\n","").split(",") for item in soup.find_all("dd")][1::2]


    #最後は"back"と表記されている。
    del author[-1]

    #URL処理
    pdf_URL = [item for item in pdf_URL if isinstance(item,str)]
    pdf_URL = ["https://openaccess.thecvf.com"+item for item in pdf_URL if '.pdf' in item and 'papers' in item]


    assert len(title)==len(pdf_URL)==len(year)==len(Meeting)==len(author),"それぞれの要素の配列数が違います。"

    paper_dict = [{
        "発行年":year[i],
        "Meeting":'Null',
        "論文名":title[i],
        "URL":pdf_URL[i]
        }
        for i in range(len(title))]

    PtA_dict = [{
        "論文ID":title[i],
        "著者ID":author[i][j]
        }
        for i in range(len(title)) #これが先
        for j in range(len(author[i])) #こっちが前のforに内包されている
        ]

    return pd.DataFrame(paper_dict),pd.DataFrame(PtA_dict)


def NeurIPS2021_scraping():
    url = "https://nips.cc/Conferences/2021/Schedule?type=Poster"
    html = requests.get(url)
    #print(html.url)
    assert html.status_code == 200,"アクセスに失敗しました"
    soup = BeautifulSoup(html.text, features="html.parser")

    #論文パラメータ作成

    #detail_urlがあるものだけ抽出することにする
    #title = [item.text for item in soup.find_all("div",class_="maincardBody")]
    detail_urls = [item.get('href') for item in soup.find_all("a",class_="btn btn-default btn-xs href_URL")]

    detail_url = detail_urls[0]

    title = []
    year = 2021
    Meeting = "NeurIPS2021"
    pdf_URL = []
    author = []

    for detail_url in detail_urls:
        detail_html = requests.get(detail_url)
        assert detail_html.status_code == 200,"アクセスに失敗しました"
        detail_soup = BeautifulSoup(detail_html.text, features="html.parser")
        sleep(0.5)

        title.append(detail_soup.find("h2",class_="note_content_title").text)
        pdf_URL.append("https://openreview.net" + detail_soup.find("a",class_="note_content_pdf").get('href'))
        author.append(detail_soup.find("div",class_="meta_row").text.split(","))

    assert len(title)==len(pdf_URL)==len(author),"それぞれの要素の配列数が違います。"

    paper_dict = [{
        "発行年":year,
        "Meeting":Meeting,
        "論文名":title[i],
        "URL":pdf_URL[i]
        }
        for i in range(len(title))]

    PtA_dict = [{
        "論文ID":title[i],
        "著者ID":author[i][j]
        }
        for i in range(len(title)) #これが先
        for j in range(len(author[i])) #こっちが前のforに内包されている
        ]

    return pd.DataFrame(paper_dict),pd.DataFrame(PtA_dict)


#df = make_paper_dict("https://openaccess.thecvf.com/CVPR2021?day=2021-06-21")
#df.to_csv(r"C:\Users\hotta_mini\Desktop\Python\database\test.csv",index=None,encoding="utf-8-sig")
#update_T_Dataset()

a,b = NeurIPS2021_scraping()

print(a)
print(b)

update_table(a,b)

"""
for i in range(len(df)):
    sql = "INSERT INTO T_論文test(発行年,Meeting,論文名, URL) VALUES({},{},'{}','{}')"
    cursor.execute(sql.format(df.iat[i,0],df.iat[i,1],df.iat[i,2],df.iat[i,3]))
    cursor.commit()
"""
