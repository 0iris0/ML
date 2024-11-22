import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 預測是否為缺陷，並比較邏輯斯回歸與clustering成效

# 匯入資料
data = pd.read_csv("manufacturing_defect_dataset.csv")
# print(data)
data_x = data.iloc[:, :-1]
data_y = data[["DefectStatus"]]
# print(data_x)

# 資料每欄確認分布狀態、除錯
print(data.describe())
print(data.isnull())

# 計算相關性、共線性

# 前處理、標準化、正規化

# 建模


# 測試

# 模型正確率達
