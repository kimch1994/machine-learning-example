import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def hyporthesis(X, theat):
    return X @ theat.T

def cost_fn(X, y, theat):
    c = np.sum(np.power(hyporthesis(X,theat) - y , 2))
    return  c / (2 * len(X))
    # 6 OR X

def gradientDescent(X, y, theta, alpha, epochs):
    cost_list = np.zeros(epochs)

    for i in range(epochs):
        theta = theta - alpha / len(X) * np.sum(((hyporthesis(X, theta) - y) * X), axis=0)
        cost_list[i] = cost_fn(X, y, theta)

    return theta, cost_list


dataframe = pd.read_csv('./insurance.csv')

dataframe = (dataframe - dataframe.mean()) / dataframe.std()

X = dataframe.iloc[:, 0:6]
X = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)

y = dataframe.iloc[:, 6:7].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
theta = np.zeros([1, len(X[0])])

learning_rate = 0.001
epochs = 1000

f_theta , cost_list = gradientDescent(X_train, y_train, theta, learning_rate, epochs)
print('Get Final THETA : ', f_theta)

f_cost = cost_fn(X, y, f_theta)
print('Get Final COST : ', f_cost)

plt.plot(np.arange(epochs), cost_list,'r')
plt.xlabel('Number of Epochs')
plt.ylabel('Cost')
plt.show()

print('테스트 데이터 개수 : ', len(X_test))
for i in range(len(X_test)):
    h = hyporthesis(X_test[i], f_theta)
    print('정답 : {} / 예측 결과 : {}'.format(y_test[i], h))
