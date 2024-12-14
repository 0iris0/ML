from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
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
list_x_names = list(data.columns)
for d in list_x_names:
    if d == "DefectStatus":
        break
    else:
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
# features = [col for col in x.columns]
# plt.figure(figsize=(15, 5))
# rows = (len(features)+2)//3
# for i, feature in enumerate(features, 1):
#     plt.subplot(rows, 3, i)
#     plt.scatter(x[feature], data["DefectStatus"], alpha=0.7, label=feature)
#     plt.xlabel(feature)
#     plt.ylabel("DefectStatus")
#     # plt.title(f"{feature} vs DefectStatus")
#     plt.grid(True)
# plt.tight_layout()  # 調整佈局
# plt.show()

# 特徵工程
# 正規化
scalar = MinMaxScaler()
x_scalar = scalar.fit_transform(x)
# 看相關性選擇特徵
cor_data = pd.concat([x, y], axis=1)
cor = pd.DataFrame(cor_data).corr()
plt.figure(figsize=(10, 10))
sns.heatmap(cor, cmap="Reds", annot=True, fmt=".3f")
plt.title("fliter defect heatmap")
# plt.show()
# 選擇保留4項特徵
fliter_x = cor[abs(cor["DefectStatus"]) > 0.1]
select_4x = list(fliter_x.index[fliter_x.index != "DefectStatus"])
# print(select_4x) #['ProductionVolume', 'DefectRate', 'QualityScore', 'MaintenanceHours']

# 前處理
data_x = data[select_4x]
data_y = data["DefectStatus"]
x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.2, random_state=6)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.2, random_state=1)

# 建模
model = LogisticRegression()
model.fit(x_train, y_train)

# 模型測試
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("混淆矩陣=", cm)  # [[ 34  78],[ 14 522]]
print("模型準確率：", round((model.score(x_test, y_test))*100, 1), "%")  # 85.8%
scores = cross_val_score(model, data_x, data_y, cv=5, scoring="accuracy")
print("交叉驗證後模型準確率：", round((scores.mean())*100, 1), "%")  # 87.3%
