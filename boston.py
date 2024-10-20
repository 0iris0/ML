from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 匯入資料
(x_train_set, y_train_set), (x_test, y_test) = boston_housing.load_data()
print(x_train_set.shape)
print(y_train_set.shape)
print(x_test.shape)
print(y_test.shape)

# 前處理
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_set, y_train_set, random_state=1)
scal = StandardScaler()
x_train = scal.fit_transform(x_train)
x_valid = scal.transform(x_valid)
x_test = scal.transform(x_test)

# 清除先前設定
keras.backend.clear_session()
# 設定seed確定結果可現性,降低初始化權重波動
np.random.seed(1)
tf.random.set_seed(1)

# 建模
model = Sequential()
model.add(Dense(units=100, activation="relu", input_shape=x_train.shape[1:]))
model.add(Dense(units=50, activation="relu"))
model.add(Dense(units=1))
print(model.summary())
weights, biases = model.layers[1].get_weights()

model.compile(loss="mse", optimizer=SGD(learning_rate=1e-3))

# 訓練模型
train = model.fit(x_train, y_train, epochs=20,
                  validation_data=(x_valid, y_valid))

# 視覺化
pd.DataFrame(train.history).plot()
plt.grid(True)
plt.show()

# 預測
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test[:3])
r2 = r2_score(y_test, y_pred)
print(f"R*2 score:{r2}")
