#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-24 下午8:55
@email: lph0729@163.com  

"""

from keras.datasets import mnist
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam


def data_pre_processing():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def build_neural_network():
    # build the first layer of the neural network
    model.add(Convolution2D(  # output_shape:[28,28,32]
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        input_shape=(28, 28, 1)
    ))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(  # output_shape:[14,14,32]
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    ))

    # build the second layer of the neural network
    model.add(Convolution2D(  # output_shape:[14,14,64]
        filters=64,
        kernel_size=(5, 5),
        padding="same"
    ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(  # output_shape:[7,7,64]
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    ))

    # build the first full layer of the neural network
    # output_shape: [1024]
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))

    # build the second full layer of the neural network
    model.add(Dense(10))
    model.add(Activation("softmax"))

    adam = Adam(lr=1e-4)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=adam,
        metrics=["accuracy"]
    )


def train_model():
    model.fit(x_train, y_train, batch_size=32, epochs=1)


def model_evaluate():
    loss, accuracy = model.evaluate(x_test, y_test)
    print("loss:", loss, "\naccuracy", accuracy)


if __name__ == '__main__':
    # 1.get data and data pre-processing
    x_train, y_train, x_test, y_test = data_pre_processing()

    # 2.build neural network
    model = Sequential()
    build_neural_network()

    # 3.train the model
    train_model()

    # 4.evaluate the model
    model_evaluate()
