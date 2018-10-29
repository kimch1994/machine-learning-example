#1 주차#
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('./dataset/train_dataset_Kaggle.csv')
train_data = train_dataset.values

test_dataset = pd.read_csv('./dataset/test_dataset_Kaggle.csv')
test_data = test_dataset.values

train_X = train_data[:, 0:-1].astype(float)
train_Y = train_data[:, [-1]]
test_X = test_data[:, 0:-1].astype(float)
test_Y = test_data[:, [-1]]

model = LinearRegression()
model.fit(train_X, train_Y)

y_predict = model.predict(test_X)

print('정답 : ', test_Y)
print('예측값 : ', y_predict)
plt.scatter(train_X, train_Y, color='red')
plt.title("Train data set")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


plt.scatter(test_X, test_Y, color='red')
plt.plot(test_X, y_predict, color='blue')
plt.title("Test data set")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

