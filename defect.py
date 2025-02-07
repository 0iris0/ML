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

# 資料轉換 #本資料集皆為數值

# 資料填補 #預設數值資料以KNN(K=3)的mean填補，類別資料以mode填補

# 資料清理(排除重複)

# 檢視異常值
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=data[feature])
#     plt.title(f"{feature}箱型圖")
#     plt.show()

# EDA
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

# 視覺化特徵與目標變數之關係
# for feature in num_features:
#     plt.figure(figsize=(10, 5))
#     plt.scatter(data[feature], data["DefectStatus"], alpha=0.2)
#     plt.title(f"{feature} vs DefectStatus")
#     plt.xlabel(feature)
#     plt.ylabel("DefectStatus")
#     plt.show()

# 排除異常值_IsolationForest
iso_forest = IsolationForest(
    n_estimators=100, contamination=0.05, random_state=11)  # 用於多特徵且能檢測非線性
pred_outlier = iso_forest.fit_predict(data.drop(columns=["DefectStatus"]))
data["is_outlier"] = pred_outlier
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
#         data = data[(data[d] >= outlier_min) & (
#             data[d] <= outlier_max)]  # 用於常態資料或近似常態
# data = data.reset_index(drop=True)# n=3240
# print(data)

# 觀察相關性
# cor = data.corr(method='spearman')#非常態
# plt.figure(figsize=(10, 10))
# sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f")
# plt.title("特徵相關性熱力圖")
# plt.show()  #相關性極低,非線性相關

# 特徵工程
# 特徵重要性
data_x = data.drop(
    columns=["DefectStatus", "SafetyIncidents"])
data_y = data["DefectStatus"]
model = xgb.XGBClassifier()
model.fit(data_x, data_y)
import_features = pd.DataFrame(
    {"Feature": data_x.columns, "Importance": model.feature_importances_})
# print(import_features)

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
# 非線性SVM
# model = SVC(kernel="rbf", C=1.0, gamma=0.1, random_state=0)
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

# 模型測試
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
# print("混淆矩陣=", cm)

# 指標分數
accuracy = round(accuracy_score(y_test, y_pred)*100, 1)
print(f"accuracy score={accuracy}")  # SVM=80.0, XGB=95.0, XGB優化=95.1
# recall_score
recall = round(recall_score(y_test, y_pred)*100, 1)
print(f"recall score={recall}")  # XGB=98.8, XGB優化=99.0
# f1_score
f1 = round(f1_score(y_test, y_pred)*100, 1)
print(f"f1 score={f1}")  # XGB=97.1, XGB優化=97.2
# auc_score
y_prob = model.predict_proba(x_test)[:, 1]
roc_auc = round(roc_auc_score(y_test, y_prob)*100, 1)
print(f"auc score={roc_auc}")  # XGB=83.7, XGB優化=85.6
# 泛化能力
scores = cross_val_score(model, x_valid,
                         y_valid, cv=10, scoring="roc_auc")
print("CV準確度=", round((scores.mean())*100, 1),
      "%")  # SVM=83.8%, XGB=90.8%, XGB優化=91.4%


# 結果分析
# 因數據分布不均，所以先看recall跟f1_score，假設公司可能希望盡量抓出高缺陷避免漏掉，可看recall=99.0
# 準確率與CV準確率相近，模型具有穩定性
# auc score=85.6表現良好=對於正負類預測良好
