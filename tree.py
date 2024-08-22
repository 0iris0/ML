import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
iris = datasets.load_iris()
x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = pd.DataFrame(iris["target"], columns=["target"])
iris_data = pd.concat([x, y], axis=1)
iris_data = iris_data[["sepal length (cm)", "petal length (cm)", "target"]]
# print(iris_data.head(3))
x_train, x_test, y_train, y_test = train_test_split(
    iris_data[["sepal length (cm)", "petal length (cm)"]], iris_data[["target"]], test_size=0.3, random_state=0)
tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
tree = tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)
print(y_pred)
y_test = y_test["target"].values
# print(y_test)
# 預測正確率
score = tree.score(x_test, y_test)
print(f'正確率={round(score*100, 1)}%')
