import shap
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from matplotlib import rc
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
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
# 因目標是二分類，初評模型可選用logistic,svm,XGBoost來比較

# 匯入資料
data = pd.read_csv("manufacturing_defect_dataset.csv")
# print(data)

# 資料概覽
# print(data.info())
# print(data.describe())
# print(data.head())
# print(data.tail())
# print(data.isnull().sum())  # 無缺失值
# print(data.nunique())

# 資料轉換

# 檢視缺失值
# sns.heatmap(data.isnull(), cmap="plasma")
# plt.title("缺失值分佈")
# plt.show()  # 無缺失值,如有缺失則進行填補

# 資料填補

# 分析目標變數,類別比例檢查
# print(data["DefectStatus"].value_counts(normalize=True)*100) #1=High Defects占84%, 0=Low Defects占16%
# sns.countplot(x="DefectStatus", data=data)
rc("font", family='Microsoft JhengHei')
plt.rcParams['axes.formatter.useoffset'] = False
# plt.title("目標變數分布")
# plt.show()

# 視覺化數值特徵
num_features = data.select_dtypes(include=["int64", "float64"]).columns
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     sns.histplot(data[feature], kde=True)
#     plt.title(f"{feature}數值分佈")
#     plt.xlabel(feature)
#     plt.ylabel("frequency")
#     plt.show()

# 視覺化特徵與目標變數之關係
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     plt.scatter(data[feature], data["DefectStatus"], alpha=0.2)
#     plt.title(f"{feature} vs DefectStatus")
#     plt.xlabel(feature)
#     plt.ylabel("DefectStatus")
#     plt.show()

# 檢定常態性_用hist看
# for i, feature in enumerate(features, 1):
#     plt.subplot(rows, 3, i)
#     plt.hist(x[feature], alpha=0.7, label=feature)
#     plt.title("is normal?")
#     plt.show()  # 非常態分佈
# 檢定常態性_用QQ-plot看
# for i, feature in enumerate(features, 1):
#     plt.subplot(rows, 3, i)
#     stats.probplot(x[feature], dist="norm", plot=plt)
#     plt.title("is normal?")
# plt.show() # 非常態分佈
# 檢定常態性_統計檢定
# for i, feature in enumerate(features, 1):
#     mean = np.mean(x[feature])
#     std = np.std(x[feature], ddof=1)
#     stat, p = stats.kstest(x[feature], "norm", args=(mean, std))
#     if p > 0.05:
#         print("資料常態分佈")
#     else:
#         print("資料非常態分佈")

# 資料探勘
# 觀察相關性
# cor = data.corr(method='spearman')#非常態
# plt.figure(figsize=(10, 10))
# sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f")
# plt.title("特徵相關性熱力圖")
# plt.show()  #相關性低,無共線性

# 檢視異常值
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=data[feature])
#     plt.title(f"{feature}箱型圖")
#     plt.show()

# 排除異常值_IsolationForest
iso_forest = IsolationForest(
    contamination=0.05, random_state=11)  # 用於高維且能檢測非線性
outliers = iso_forest.fit_predict(data.drop(columns=["DefectStatus"]))
data["is_outlier"] = outliers
data = data[data["is_outlier"] == 1].drop(columns=["is_outlier"])  # n=3078
# 排除異常值_modified_z_score
# for col in num_features:
#     median=data[col].median()
#     mad=median_absolute_deviation(data[col])#用於近似常態,維度小,數據量少,異常值較少
#     data["modified_z_score"]=0.6745*(data[col]-median)/mad
# data=data[data["modified_z_score"].abs()<3.5].drop(columns=["modified_z_score"])
# 排除異常值_1.5IQR
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
data_x = data.drop(columns=["DefectStatus"])
data_y = data["DefectStatus"]
x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.2, random_state=6)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.2, random_state=11)
# 正規化
# scaler = MinMaxScaler()
# x_train_scalered = scaler.fit_transform(x_train)  # 用XGBoost不用正規化
# x_valid_scalered = scaler.transform(x_valid)
# x_test_scalered = scaler.transform(x_test)

# 特徵重要性
model = xgb.XGBClassifier()
model.fit(data_x, data_y)
import_features = pd.DataFrame(
    {"Feature": data_x.columns, "Importance": model.feature_importances_})
# print(import_features)
# 創建 SHAP 解釋器
# explainer = shap.Explainer(model)
# shap_values = explainer(x_test)
# print(shap_values)

# 建模
# 非線性SVM
# model = SVC(kernel="rbf", C=1.0, gamma=0.1, random_state=0)
# model.fit(x_train_scalered, y_train)
# XGBoost
model = XGBClassifier()
model.fit(x_train, y_train)

# XGB超參數優化
# param_dist = {
#     'n_estimators': np.arange(50, 300),
#     'max_depth': [3, 5, 7, 9],
#     'learning_rate': np.linspace(0.01, 0.3, 10),
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }
# rscv = RandomizedSearchCV(
#     estimator=model,
#     param_distributions=param_dist,
#     n_iter=30,
#     scoring="accuracy",
#     cv=10
# )
# rscv.fit(x_valid, y_valid)
# print("優化最佳參數:", rscv.best_params_)
# print("優化最佳準確率:", round((rscv.best_score_)*100, 1), "%")
# model = rscv.best_estimator_

# 模型測試
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
# print("混淆矩陣=", cm)
# 準確率
accuracy = round(accuracy_score(y_test, y_pred)*100, 1)
print(f"準確率={accuracy}")  # SVM=80.0%, XGB=95.3%, XGB優化=95.1%
# recall_score
recall = round(recall_score(y_test, y_pred)*100, 1)
print(f"recall分數={recall}%")  # XGB=99.2%, XGB優化=99.2%
# f1_score
f1 = round(f1_score(y_test, y_pred)*100, 1)
print(f"f1 score={f1}%")  # XGB=97.3%, XGB優化=97.2%
# auc_score
y_prob = model.predict_proba(x_test)[:, 1]
roc_auc = round(roc_auc_score(y_test, y_prob)*100, 1)
print(f"auc分數={roc_auc}%")  # XGB=84.5%, XGB優化=86.2%
# 泛化能力
scores = cross_val_score(model, x_valid,
                         y_valid, cv=10, scoring="roc_auc")
print("CV準確率=", round((scores.mean())*100, 1),
      "%")  # SVM=83.8%, XGB=94.9%, XGB優化=95.7%


# 結果分析
# 因數據分布不均，所以先看recall跟f1_score，因看資料分布擁有較多高缺陷，所以猜測公司可能希望盡量抓出疑似缺陷避免漏掉，因此主看recall
# 準確率與CV準確率相近，模型具有穩定性
# auc分數86.7%表現良好=對於正負類預測良好，但還可改進，ex:增加低缺陷資料、再正規化、調整測試集數量
# 看要不要調整分類閾值
