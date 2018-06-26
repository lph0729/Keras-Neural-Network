#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-25 下午3:05
@email: lph0729@163.com  

"""
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist

TIME_STEPS = 28
INPUT_SIZE = 28
OUTPUT_SIZE = 10
CELL_SIZE = 50


def data_processing():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28) / 255.
    x_test = x_test.reshape(-1, 28, 28) / 255.

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def build_recurrent_neural_network():
    # RNN cell
    model.add(SimpleRNN(
        CELL_SIZE,
        batch_input_shape=(None, TIME_STEPS, INPUT_SIZE)
    ))

    # output layer
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation("softmax"))

    # optimizer
    adam = Adam()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=adam,
        metrics=["accuracy"]
    )


def model_train():
    BATCH_SIZE = 50
    BATCH_INDEX = 0
    for step in range(10000):
        x_batch = x_train[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :, :]
        y_batch = y_train[BATCH_INDEX:BATCH_SIZE + BATCH_INDEX, :]

        model.train_on_batch(x_batch, y_batch)

        BATCH_INDEX += BATCH_SIZE

        BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX

        # 4.evaluate RNN model
        model_evaluate(step)


def model_evaluate(step):
    if step % 100 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print("test-cost:", cost, "test-accuracy:", accuracy)


def model_predict_accuracy():
    for step in range(1000):
        if step % 100 == 0:
            y_predict = model.predict(x_test, batch_size=y_test.shape[0])
            print("y_predict:", y_predict)


if __name__ == '__main__':
    # 1.get data and data pro-precessing
    x_train, y_train, x_test, y_test = data_processing()

    # 2.build the RNN model
    model = Sequential()
    build_recurrent_neural_network()

    # 3.train RNN model
    model_train()
    # 4.evaluate RNN model

    # 5.predict the accuracy of the model
    model_predict_accuracy()
