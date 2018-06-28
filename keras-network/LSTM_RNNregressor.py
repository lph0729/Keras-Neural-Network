#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-27 下午5:02
@email: lph0729@163.com  

"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np

CELL_SIZE = 10
BATCH_START = 0
BATCH_SIZE = 20
TIME_STEPS = 20
INPUT_SIZE = 1
OUTPUT_SIZE = 1


def generate_model_data():
    global BATCH_START
    # for temp in range(10):
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE).reshape(BATCH_SIZE, TIME_STEPS) / (10 * np.pi)

    # print(xs)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs, seq, "r", xs, res, "b--")
    # plt.show()

    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


def build_lstm_rnn():
    model.add(LSTM(
        CELL_SIZE,
        batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
        return_sequences=True,
        stateful=True
    ))

    model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
    adam = Adam()
    model.compile(
        loss="mse",
        optimizer=adam
    )


def rnn_training_model():
    for step in range(500):
        x_batch, y_batch, xs = generate_model_data()
        cost = model.train_on_batch(x_batch, y_batch)
        pred = model.predict(x_batch, BATCH_SIZE)
        plt.plot(xs[0, :], y_batch[0], "r", xs[0, :], pred.flatten()[:TIME_STEPS], "b--")
        plt.ylim(-1.2, 1.2)
        plt.draw()
        plt.pause(0.1)

        if step % 10 == 0:
            print("step:", step, "cost:", cost)


if __name__ == '__main__':
    # 1.generate model data

    # print(x_batch[0, :].shape)
    # print(y_batch[0].flatten().shape)

    # 2.build LSTM RNN
    model = Sequential()
    build_lstm_rnn()

    # 3.start training the model
    rnn_training_model()
