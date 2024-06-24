# 探討以專業影評人打分(Meta_score)預測IMBD分數(IMBD_Rating)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
dataset = pd.read_csv("imdb_top_1000.csv")

# 資料前處理
"""排除沒有Meta_score的資料,將IMBD分類為好評(>=7分)與差評(<7分)"""
fliter = dataset.dropna(subset=["Meta_score"])
df = pd.DataFrame(fliter)
ab = {True: "good", False: "bad"}
df["IMDB_Rating"] = df["IMDB_Rating"] >= 7.0
df["IMDB_Rating"] = df["IMDB_Rating"].map(ab)
x = df["Meta_score"].values.reshape(-1, 1)  # 轉二維也可用df[["Meta_score"]]
y = df["IMDB_Rating"]

# 拆分訓練集與測試集(7:3)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# 訓練模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# 預測
pred = knn.predict(x_test).tolist()
acc = accuracy_score(y_test, pred)
print("準度:", acc) #準度:1.0

from sklearn.model_selection import cross_val_score #交叉驗證
k_range=range(2,12)
k_scores=[]
for k in k_range:
    knn_model=KNeighborsClassifier(n_neighbors=k)
    accuracy=cross_val_score(knn_model,x_train,y_train,cv=10,scoring="accuracy")
    print("k=",str(k), "accuracy=",str(accuracy.mean())
    k_scores.append(accuracy.mean())
    print("Best k=",max(k_scores))
          
# 視覺化
plt.rc('font', family='Microsoft JhengHei')
plt.plot(x_train, knn.predict(x_train), color="blue")
plt.title("專業影評人評分 vs IMBD評價")
plt.xlabel("專業影評人評分")
plt.ylabel("IMBD評價")
plt.show()
