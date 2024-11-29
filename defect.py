from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 預測是否為缺陷，並比較羅吉斯回歸與clustering成效

# 匯入資料
data = pd.read_csv("manufacturing_defect_dataset.csv")
# print(data)

# 資料每欄確認分布狀態、除錯(排除異常值、填補)
# print(data.info())
# print(data.describe())
# print(data.isnull())
# print(data.head())
list_x_names = list(data.columns)
for d in list_x_names:
    per25 = np.percentile(data[d], 25)
    per75 = np.percentile(data[d], 75)
    IQR = per75-per25
    outlier_max = per75+1.5*IQR
    outlier_min = per25-1.5*IQR
    data = data[(data[d] >= outlier_min) & (data[d] <= outlier_max)]
data = data.reset_index(drop=True)
# print(data)


# 視覺化
x = data.iloc[:, :-1]
y = data[["DefectStatus"]]
# print(x)
features = [col for col in x.columns]
plt.figure(figsize=(15, 5))
rows = len(features)//3
for i, feature in enumerate(features, 1):
    plt.subplot(rows, 3, i)
    plt.scatter(x[feature], data["DefectStatus"], alpha=0.7, label=feature)
    plt.xlabel(feature)
    plt.ylabel("DefectStatus")
    plt.title(f"{feature} vs DefectStatus")
    plt.grid(True)
    # plt.show()

# 計算相關性、共線性

# a = data.corr()
# print(a)
# 前處理、標準化、正規化

# 建模


# 測試

# 模型正確率達
