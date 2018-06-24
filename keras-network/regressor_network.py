#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-23 下午6:53
@email: lph0729@163.com  

"""
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import time


def generate_model_data():
    np.random.seed(45)
    x = np.linspace(-1, 1, 200)
    np.random.shuffle(x)
    bias = np.random.normal(0, 0.05, (200,))

    y = 0.5 * x + 2 + bias

    # plt.scatter(x, y)
    # plt.show()

    x_train, y_train = x[:160], y[:160]
    x_test, y_test = x[160:], y[160:]

    return x_train, y_train, x_test, y_test


def build_nerual_network():
    model.add(Dense(input_dim=1, output_dim=1))
    # mse(均方误差)，常用的目标函数，公式为((y_pred-y_true)**2).mean()   sgd:梯度下降函数
    model.compile(loss="mse", optimizer="sgd")


def model_train(x_train, y_train):
    for step in range(10000):
        cost = model.train_on_batch(x_train, y_train)
        if step % 100 == 0:
            print("train_cost:", cost)


def model_evaluate(x_test, y_test):
    cost = model.evaluate(x_test, y_test, batch_size=40)
    print("test_cost:", cost)
    w, b = model.layers[0].get_weights()
    print("weight=", w, "\nbias=", b)
    time.sleep(3600)


def model_show(x_test, y_test):
    y_pred = model.predict(x_test)
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred)
    plt.show()


model = Sequential()  # Sequential model is a linear stack of layers

# 1.generate some data
x_train, y_train, x_test, y_test = generate_model_data()

# 2.buid a neural network 、choose loss function and optimizing method
build_nerual_network()

# 3.training model
model_train(x_train, y_train)

# 4.test model
model_evaluate(x_test, y_test)

# 5.plotting the prediction
model_show(x_test, y_test)
