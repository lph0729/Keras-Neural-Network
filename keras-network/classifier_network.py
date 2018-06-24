#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-24 上午11:27
@email: lph0729@163.com  

"""
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


def data_pre_processing():
    # download the mnist to the path "~/.keras/datasets" if it is the first time to be called

    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 主要是将标签纸转化为one-hot编码
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# 1.get data and pre-processing
x_train, y_train, x_test, y_test = data_pre_processing()

# 2.build neural network
model = Sequential([
    Dense(32, input_dim=784),
    Activation("relu"),
    Dense(10),
    Activation("softmax")
])
# 加速下降算法RMSprop，全称root mean square prop:
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(
    loss="categorical_crossentropy",
    optimizer=rmsprop,
    metrics=["accuracy"]
)

# 3.train the model
print("-------开始训练模型-----------------")
model.fit(x_train, y_train, batch_size=32, epochs=2)

# 4.evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("test_loss:", loss)
print("test_accuracy:", accuracy)
