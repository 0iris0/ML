# 探討以專業影評人打分(Meta_score)預測IMBD分數(IMBD_Rating)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("imdb_top_1000.csv")

# 資料前處理
"""刪除沒有Meta_score的資料。不採用填補方式及刪除離群值,因沒有此欄資料評分條件或評分因素可以列入考量(IMBD網站不公開)
考量電影類型可能也會影響評分,一種電影不只屬於一種類型,若用mean或median會降低x影響"""
fliter = dataset.dropna(subset=["Meta_score"])
df = pd.DataFrame(fliter)
# print(df.shape)
x = df["Meta_score"].values.reshape(-1, 1)  # 轉二維也可用df[["Meta_score"]]
y = df["IMDB_Rating"]

# 拆分訓練集與測試集(7:3)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# 訓練模型
reg = LinearRegression()
reg.fit(x_train, y_train)

# 預測
inter = reg.intercept_
coe = reg.coef_
print("截距:", inter)  # 截距: 7.546476707567622
print("係數:", coe)  # 係數: [0.00494949]

score = reg.score(x_test, y_test)
print("準度:", str(round(score*100, 2)), "%")  # 準度: 11.07 %

# 視覺化
plt.rc('font', family='Microsoft JhengHei')
plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="blue")
plt.title("專業影評人評分 vs IMBD分數")
plt.xlabel("專業影評人評分")
plt.ylabel("IMBD分數")
plt.show()

# 結果分析
"""相關係數僅約0.005,雖然係數與相關性不見得是正比,但專業影評人打分每改變1分對IMBD分數影響極小"""

# 進階探討方向
"""
1.觀眾評分是否受到促銷方式、公關票等影響
2.高票房不等於好片,專業影評人評分標準如有釋出,可以透過評分標準不同特徵子集合針對IMBD分數與票房做預測
"""
