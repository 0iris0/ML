from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
data = pd.read_csv("Salary_Data2.csv")
# print(data)
data["EducationLevel"] = data["EducationLevel"].map(
    {"高中以下": 0, "大學": 1, "碩士以上": 2})
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data[["City"]])
city_encoded = onehot_encoder.transform(data[["City"]]).toarray()
data[["CityA", "CityB", "CityC"]] = city_encoded
data = data.drop(["City", "CityC"], axis=1)
x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
w = np.array([1, 2, 3, 4])
b = 1
y_pred = (x_train*w).sum(axis=1)+b
cost = ((y_train-y_pred)**2).mean()


def compu_cost(x, y, w, b):
    y_pred = (x*w).sum(axis=1)+b
    cost = ((y-y_pred)**2).mean()
    return cost


def compu_grad(x, y, w, b):
    y_pred = (x*w).sum(axis=1)+b
    b_gradient = (y_pred-y).mean()
    w_gradient = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        w_gradient[i] = (x[:, i]*(y_pred-y)).mean()
    # w1_gradient = (x_train[:, 0]*(y_pred-y_train)).mean()
    # w2_gradient = (x_train[:, 1]*(y_pred-y_train)).mean()
    # w3_gradient = (x_train[:, 2]*(y_pred-y_train)).mean()
    # w4_gradient = (x_train[:, 3]*(y_pred-y_train)).mean()
    return w_gradient, b_gradient


w = 0
b = 0
learning_rate = 0.001

np.set_printoptions(formatter={"float": ":.2e".format})


def gradient_descent(x, y, w_init, b_init, learning_rate, cost_func, grad_func, run_iter, p_iter=1000):
    c_hist = []
    w_hist = []
    b_hist = []
    w = w_init
    b = b_init
    for i in range(run_iter):
        w_gradient, b_gradient = grad_func(x_train, y_train, w, b)
        w = w-w_gradient*learning_rate
        b = b-b_gradient*learning_rate
        cost = cost_func(x, y, w, b)
        c_hist.append(cost)
        w_hist.append(w)
        b_hist.append(b)
        if i % p_iter == 0:
            print(
                f"iter第{i}次:Cost={cost:.2f},w={w},b={b:.2f},w_gradient={w_gradient},b_gradient={b_gradient:.2f}")
    return w, b, w_hist, b_hist, c_hist


w_init = 0
b_init = 0
learning_rate = 0.001
run_iter = 10000
w_fin, b_fin, w_hist, b_hist, c_hist = gradient_descent(x, y, w_init, b_init, learning_rate,
                                                        compu_cost, compu_grad, run_iter, p_iter=1000)
print(w_fin, b_fin, w_hist, b_hist, c_hist)

print((w_fin*x_test).sum(axis=1)+b_fin)
a = pd.DataFrame({
    "y_pred": y_pred,
    "y_test": y_test
})
print(a)
