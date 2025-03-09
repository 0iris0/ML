from sklearn.linear_model import LogisticRegression
import math
import shap
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, roc_curve
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
# 因目標是二分類，初評模型可選用logistic,svm,樹模型來比較

# 匯入資料
data = pd.read_csv("manufacturing_defect_dataset.csv")
# print(data)

# 資料概覽
# print(data.info())
# print(data.describe())
# print(data.head())
# print(data.tail())
# print(data.isnull().sum())  # 無缺失值
# 視覺化缺失值分佈
# sns.heatmap(data.isnull(), cmap="plasma")
# plt.title("缺失值分佈")
# plt.show()
# print(data.nunique())

# 資料清理(排除重複)

# EDA
# 分析目標變數,類別比例檢查
# print(data["DefectStatus"].value_counts(normalize=True)*100) #1=High Defects占84%, 0=Low Defects占16%
# sns.countplot(x="DefectStatus", data=data)
rc("font", family='Microsoft JhengHei')
plt.rcParams['axes.formatter.useoffset'] = False
# plt.title("目標變數分布")
# plt.show()

# 起始資料非常態進行轉換後仍為非常態
nor_trans_data_x = data.drop(columns=["DefectStatus"])
nor_trans_data_x = np.log1p(nor_trans_data_x)
# print(nor_trans_data_x)

# 視覺化數值特徵
num_features = data.select_dtypes(include=["int64", "float64"]).columns
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     sns.histplot(data[feature], kde=True)
#     plt.title(f"{feature}數值分佈")
#     plt.xlabel(feature)
#     plt.ylabel("frequency")
#     plt.show()

# 檢定常態性_用hist看
rows = len(num_features)//3+1
# for i, feature in enumerate(num_features, 1):
#     plt.subplot(rows, 3, i)
#     plt.hist(data[feature], alpha=0.7, label=feature)
#     plt.title("is normal?")
#     plt.show()  # 非常態分佈
# 檢定常態性_統計檢定
# for i, feature in enumerate(num_features, 1):
#     mean = np.mean(data[feature])
#     std = np.std(data[feature], ddof=1)
#     stat, p = stats.kstest(data[feature], "norm", args=(mean, std))
#     if p > 0.05:
#         print("資料常態分佈")
#     else:
#         print("資料非常態分佈")

# 視覺化特徵與目標變數之關係
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     plt.scatter(data[feature], data["DefectStatus"], alpha=0.2)
#     plt.title(f"{feature} vs DefectStatus")
#     plt.xlabel(feature)
#     plt.ylabel("DefectStatus")
#     plt.show()

# 檢視異常值
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=data[feature])
#     plt.title(f"{feature}箱型圖")
#     plt.show()

# 排除異常值_IsolationForest
iso_forest = IsolationForest(
    n_estimators=100, contamination=0.05, random_state=11)  # 用於多特徵且能檢測非線性
pred_outlier = iso_forest.fit_predict(data.drop(columns=["DefectStatus"]))
data["is_outlier"] = pred_outlier
data = data[data["is_outlier"] == 1].drop(columns=["is_outlier"])  # n=3078
# print(data)

# 觀察相關性
# cor = data.corr(method='spearman')#非常態
# plt.figure(figsize=(10, 10))
# sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f")
# plt.title("特徵相關性熱力圖")
# plt.show()  #相關性極低,非線性相關

# 特徵篩選
data_x = data.drop(columns=["DefectStatus", "SafetyIncidents"])
data_y = data["DefectStatus"]
model = xgb.XGBClassifier()
model.fit(data_x, data_y)
import_features = pd.DataFrame(
    {"Feature": data_x.columns, "Importance": model.feature_importances_})
print(import_features.sort_values("Importance", ascending=False))

# 前處理
x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.2, random_state=6)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.2, random_state=11)
# 正規化
# scaler = MinMaxScaler()
# x_train_scalered = scaler.fit_transform(x_train)  # 用XGBoost不用正規化
# x_valid_scalered = scaler.transform(x_valid)
# x_test_scalered = scaler.transform(x_test)

# 建模
# logistic
# model = LogisticRegression(class_weight="balanced")
# model.fit(x_train, y_train)

# 非線性SVM
# model = SVC(kernel="rbf", C=1.0, gamma=0.1, random_state=0, probability=True)
# model.fit(x_train_scalered, y_train)

# XGBoost
model = XGBClassifier(scale_pos_weight=y_train.value_counts()[
                      0]/y_train.value_counts()[1])
model.fit(x_train, y_train)
# XGB超參數優化
param_dist = {
    'n_estimators': np.arange(50, 100, 300),
    'max_depth': [3, 5, 7, 9],
    'learning_rate': np.linspace(0.05, 0.2),
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
# print("優化最佳參數:", rscv.best_params_)
# print("優化最佳準確率:", round((rscv.best_score_)*100, 1), "%")
model = rscv.best_estimator_

# 創建 SHAP 解釋器
explainer = shap.Explainer(model)
shap_values = explainer(x_train)
# print(shap_values)
# shap.summary_plot(shap_values, x_train)

# 模型泛化能力
scores = cross_val_score(model, x_valid,
                         y_valid, cv=10, scoring="roc_auc")
# print("CV準確度=", round((scores.mean())*100, 1),"%")  # logistic=85.4%, SVM=83.8%, XGB=90.5%, XGB優化=92.4%

# 模型測試
y_pred = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
# print("混淆矩陣=", cm)

# 指標分數
# accuracy
accuracy = round(accuracy_score(y_test, y_pred)*100, 1)
print(f"accuracy score={accuracy}")  # logistic=84.3, XGB=94.8, XGB優化=95.3
# precision
precision = round(precision_score(y_test, y_pred)*100, 1)
print(f"precision score={precision}")  # logistic=85.9, XGB=95.5, XGB優化=95.4
# recall_score
recall = round(recall_score(y_test, y_pred)*100, 1)
print(f"recall score={recall}")  # logistic=97.3 , XGB=98.5, XGB優化=99.2
# f1_score
f1 = round(f1_score(y_test, y_pred)*100, 1)
print(f"f1 score={f1}")  # logistic=91.2 , XGB=97.0, XGB優化=97.3
# auc_score
roc_auc = round(roc_auc_score(y_test, y_pred_prob)*100, 1)
print(f"auc score={roc_auc}")  # logistic=77.7, XGB=84.7, XGB優化=85.6

# ROC曲線圖
# fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color="b", label=f"ROC Curve(AUC={roc_auc:.2f})")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()

# 結果分析
# 因數據分布不均，所以先看recall跟f1_score，假設公司可能希望盡量抓出高缺陷避免漏掉，可看recall=99.2
# CV準確率達90%以上，模型具有穩定性
# auc score表現良好=對於正負類預測良好
