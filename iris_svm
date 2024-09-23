import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()
# print(iris.DESCR)
x = iris.data
y = iris.target
x = x[:, 2:]
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=87)
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
# plt.show()
clf = SVC()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_predict)
# plt.show()
# plt.scatter(x_test[:, 0], x_test[:, 1], c=y_predict-y_test)
# plt.show()
x1, x2 = np.meshgrid(np.arange(0, 7, 0.02), np.arange(0, 3, 0.02))
z = clf.predict(np.c_[x1.ravel(), x2.ravel()])
z = z.reshape(x1.shape)
plt.contourf(x1, x2, z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y_train)
plt.show()
