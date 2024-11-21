from sklearn.metrics import confusion_matrix
from keras.layers import Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import tensorflow as tf
import keras

# 匯入資料
(x_train_set, y_train_set), (x_test, y_test) = fashion_mnist.load_data()
print(x_train_set.shape)
print(y_train_set.shape)
print(x_test.shape)
print(y_test.shape)

i = 0
print(y_train_set[i])
plt.imshow(x_train_set[i], cmap="binary")
plt.show()

# 前處理
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_set, y_train_set, random_state=1)
x_train = x_train/255.0
x_valid = x_valid/255.0
x_test = x_test/255.0

# 清除設定
keras.backend.clear_session()
np.random.seed(1)
tf.random.set_seed(1)

# 建模
model = Sequential([
    Flatten(input_shape=x_train.shape[1:]),
    Dense(units=300, activation="relu"),
    Dense(units=200, activation="relu"),
    Dense(units=100, activation="relu"),
    Dense(units=10, activation="softmax")
])
print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", metrics=["accuracy"])

# 訓練模型
train = model.fit(x_train, y_train, epochs=20,
                  validation_data=(x_valid, y_valid))

# 視覺化
pd.DataFrame(train.history).plot()
plt.grid()
plt.show()

# 預測
model.evaluate(x_test, y_test)
y_proba = model.predict(x_test)
y_proba[:3].round(2)
y_pred = np.argmax(y_proba, axis=1)
confusion_matrix(y_test, y_pred)
