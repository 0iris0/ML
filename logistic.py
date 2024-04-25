from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
data = pd.read_csv("Diabetes_Data.csv")
# print(data)

# 資料前處理
data["Gender"] = data["Gender"].map({"男生": 0, "女生": 1})

# 拆分訓練集與測試集(8:2)
x = data[["Age", "Weight", "BloodSugar", "Gender"]]
y = data[["Diabetes"]]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
# print(x_train, x_test, y_train, y_test)

# 特徵縮放
scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
# print(x_train, x_test)

# 模型訓練
w = np.array([1, 2, 3, 4])
b = 1
z = (w*x_train).sum(axis=1)+b


def sigmoid(z):
    return 1/(1+np.exp(-z))


# print(sigmoid(z))
log = linear_model.LogisticRegression()
log.fit(x_train, y_train)
print(log.predict(x_test))
print(log.predict_proba(x_test))

# 預測
print(log.score(x_test, y_test))  # 預測力88.75%
