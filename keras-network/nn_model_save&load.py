#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-27 下午11:05
@email: lph0729@163.com  

"""
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np


def generate_data():
    x = np.linspace(-1, 1, 200)
    np.random.shuffle(x)
    y = 0.5 * x + 2 + np.random.normal(loc=0.0, scale=0.01, size=(200,))

    x_train, y_train = x[:160], y[:160]
    x_test, y_test = x[160:], y[160:]

    return x_train, y_train, x_test, y_test


def build_neural_network():
    model.add(Dense(
        units=1,
        input_dim=1
    ))

    model.compile(
        loss="mse",
        optimizer="sgd"
    )


def train_model():
    global model
    for step in range(1000):
        cost = model.train_on_batch(x_train, y_train)

    print("test before save:", model.predict(x_train[:2]))
    model.save("./model_file/linear_model.h5")   # save as HDF5 file

    # load model
    model = load_model("./model_file/linear_model.h5")
    print("test after load:", model.predict(x_train[:2]))

    """
    # save and load weights
    model.save_weights("model_weights.h5")
    model.load_weights("model_weights.h5")
    
    # save and load fresh network without trained weights
    from keras.models import model_from_json
    json_string = model.to_json()
    model = model_from_json(json_string)
    """


if __name__ == '__main__':
    # 1.generate some data
    x_train, y_train, x_test, y_test = generate_data()

    # 2.build nn model
    model = Sequential()
    build_neural_network()

    # 3.train_model
    train_model()
