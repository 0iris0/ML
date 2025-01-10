from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# 預測是否為缺陷
# 初評模型可選用logistic,svm,XGBoost來比較

# 匯入資料
data = pd.read_csv("manufacturing_defect_dataset.csv")
# print(data)

# 資料概覽
# print(data.info())
# print(data.describe())
# print(data.head())
# print(data.tail())
# print(data.isnull().sum())
# print(data.nunique())

#分析目標變數
sns.countplot(x="DefectStatus",data=data)
plt.rc("font",family="Microsoft JhengHei")
plt.title("目標變數類別分佈")
plt.show()
#類別比例檢查
print(data["DefectStatus"].value_counts(normalize=True)*100)

#分析各個數值特徵
num_features=data.select_dtypes(include=["int64","float64"]).columns
for feature in num_features:
    plt.figure(figsize=(10,5))
    sns.hisplot(data[feature],kde=True)
    plt.title(f"{feature}數值分佈")
    plt.xlabel(feature)
    plt.ylabel("frequency")
    plt.show()
# 檢定常態性
# 用hist看
# for i, feature in enumerate(features, 1):
#     plt.subplot(rows, 3, i)
#     plt.hist(x[feature], alpha=0.7, label=feature)
#     plt.title("is normal?")
# plt.show()  # 非常態分佈
# 用QQ-plot看
# for i, feature in enumerate(features, 1):
#     plt.subplot(rows, 3, i)
#     stats.probplot(x[feature], dist="norm", plot=plt)
#     plt.title("is normal?")
# plt.show() # 非常態分佈
# 統計檢定
# for i, feature in enumerate(features, 1):
#     mean = np.mean(x[feature])
#     std = np.std(x[feature], ddof=1)
#     stat, p = stats.kstest(x[feature], "norm", args=(mean, std))
#     if p > 0.05:
#         print("資料常態分佈")
#     else:
#         print("資料非常態分佈")
#檢查異常值
for feature in num_features:
    plt.figure(figsize=(10,5))
    sns.boxplot(x=data[feature])
    plt.title(f"{feature}箱型圖")
    plt.show()
sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
plt.title("缺失值分佈")
plt.show()

#視覺化特徵與目標變數之關係
for feature in num_features:
    plt.figure(figsize=(10,5))
    plt.plot(x=data[feature],y=data["DefectStatus"])
    plt.title(f"{feature} vs DefectStatus")
    plt.xlabel(feature)
    plt.ylabel("DefectStatus")
    plt.show()
#觀察相關性
cor = data.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("特徵相關性熱力圖")
# plt.show()# 特徵皆非線性
#檢定共線性VIF


#分析各個類別特徵
categ_features=data.select_dtypes(include=["object","category"]).columns
for categ in categ_features:
    plt.figure(figsize=(10,5))
    sns.countplot(x=categ,data=data=data)
    plt.title(f"{feature}類別分佈")
    plt.show()
#視覺化特徵與目標變數之關係
    plt.figure(figsize=(10,5))
    plt.barplot(x=categ,y='DefectStatus',data=data)
    plt.title(f"{categ} vs DefectStatus"")
    plt.xlabel(categ)
    plt.ylabel("DefectStatus")
    plt.show()

#資料探勘
#排除異常值_IsolationForest
from sklearn.ensemble import IsolationForest
iso_forest=IsolationForest(contamination=0.05,random_state=11)#用於高維且能檢測非線性
outliers=iso_forest.fit_predict(data)
data["is_outlier"]=outliers
data=data[data["is_outlier"]==1].drop(columns=["is_outlier"])
#排除異常值_modified_z_score
from scipy.stats import median_absolute_deviation
# for col in num_features:
#     median=data[col].median()
#     mad=median_absolute_deviation(data[col])#用於近似常態,維度小,數據量少,異常值較少
#     data["modified_z_score"]=0.6745*(data[col]-median)/mad
# data=data[data["modified_z_score"].abs()<3.5].drop(columns=["modified_z_score"])
# #排除異常值_1.5IQR
# list_x_names = list(data.columns)
# for d in list_x_names:
#     if d == "DefectStatus":
#         break
#     else:
#         per25 = np.percentile(data[d], 25)
#         per75 = np.percentile(data[d], 75)
#         IQR = per75-per25
#         outlier_max = per75+1.5*IQR
#         outlier_min = per25-1.5*IQR
#         data = data[(data[d] >= outlier_min) & (data[d] <= outlier_max)]#用於常態資料
# data = data.reset_index(drop=True)
# print(data)

# 前處理
data_x = data[:,-1]
data_y = data["DefectStatus"]
x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.2, random_state=6)
# 正規化
# scaler = MinMaxScaler()
# x_train_scalered = scaler.fit_transform(x_train)#用XGBoost不用
# x_test_scalered = scaler.fit_transform(x_test)

# 建模
# 非線性SVM
# model = SVC(kernel="rbf", C=1.0, gamma=0.1, random_state=0)
# model.fit(x_train_scalered, y_train)
# XGBoost
model = XGBClassifier()
model.fit(x_train, y_train)

# XGB超參數優化
param_dist = {
    'n_estimators': np.arange(50, 300),
    'max_depth': [3, 5, 7, 9],
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
rscv = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    scoring="accuracy",
    cv=10
)
rscv.fit(x_train, y_train)
print("優化最佳參數:", rscv.best_params_)
print("優化最佳準確率:", round((rscv.best_score_)*100, 1), "%")  # XGB優化=96.3%
model = rscv.best_estimator_

# 模型測試
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
# SVM=[[2 110],[0 536]], XGB[[85 27],[4 532]], XGB優化=[[ 84  28],[  4 532]]
print("混淆矩陣=", cm)
accuracy = round(accuracy_score(y_test, y_pred), 1)
print("預測準確率:", round((accuracy*100), 1),
      "%")  # SVM=80.0%, XGB=100.0%, XGB優化=100.0%
print("測試集x準確率:", round((model.score(x_test, y_test))*100, 1),
      "%")  # SVM=83.0%, XGB=95.2%, XGB優化=95.1%
scores = cross_val_score(model, data_x,
                         data_y, cv=10, scoring="accuracy")
print("交叉驗證準確率：", round((scores.mean())*100, 1),
      "%")  # SVM=84.8%, XGB=95.7%, XGB優化=96.1%

# 採用XGB產生的模型，因準確率達96.1%最高
