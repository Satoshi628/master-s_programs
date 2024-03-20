import pyodbc
import pandas as pd
import numpy as np
text = "what's this"
text = text.replace("'","''")
print(text)
print(pyodbc.drivers())
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\Users\hotta_mini\Desktop\Python\database\論文.accdb;'
    )
conn = pyodbc.connect(conn_str)

# データベースの指定したテーブルのデータをすべて抽出します
sql = 'SELECT * FROM T_論文'
df = pd.read_sql(sql, conn)
print(df['手法分野'])
print(df['手法分野'].isnull())
"""
df =df.replace(np.nan,'Null')
Meeting = df['手法分野'][3]
print(Meeting)
cursor = conn.cursor()

sql = "INSERT INTO T_論文(論文ID,Meeting,論文名, URL) VALUES(0,{},'{}', 'てすとURL')".format(Meeting,text)
cursor.execute(sql)
cursor.commit()
cursor.close()
"""
# データベースの接続を閉じます
conn.close()
