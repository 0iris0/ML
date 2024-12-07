from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 預測是否為缺陷(使用logistic regression)

# 匯入資料
data = pd.read_csv("manufacturing_defect_dataset.csv")
# print(data)

# 資料每欄確認分布狀態、除錯、排除異常值、填補...
# print(data.info())
# print(data.describe())
# print(data.isnull())
# print(data.head())
data_x = data.iloc[:, :-1]
list_x_names = list(data_x.columns)
for d in list_x_names:
    per25 = np.percentile(data_x[d], 25)
    per75 = np.percentile(data_x[d], 75)
    IQR = per75-per25
    outlier_max = per75+1.5*IQR
    outlier_min = per25-1.5*IQR
    data_x = data_x[(data_x[d] >= outlier_min) & (data_x[d] <= outlier_max)]
data_x = data_x.reset_index(drop=True)
# print(data_x)

# 視覺化
x = data_x.iloc[:]
y = data[["DefectStatus"]]
# print(x)
features = [col for col in x.columns]
plt.figure(figsize=(15, 5))
rows = (len(features)+2)//3
for i, feature in enumerate(features, 1):
    plt.subplot(rows, 3, i)
    plt.scatter(x[feature], data["DefectStatus"], alpha=0.7, label=feature)
    plt.xlabel(feature)
    plt.ylabel("DefectStatus")
    # plt.title(f"{feature} vs DefectStatus")
    plt.grid(True)
plt.tight_layout()  # 調整佈局
# plt.show()

# 特徵工程
# 標準化
clf = LogisticRegression()
scalar = StandardScaler()
x_scalar = scalar.fit_transform(x)
# print(x_scalar)

# 特徵選擇，看特徵相關性(heatmap)


# 建模


# 測試

# 模型正確率達
