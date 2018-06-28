#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-28 下午9:00
@email: lph0729@163.com  

"""

from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from matplotlib import pyplot as plt

# 1. mnist datasets pre-processing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255 - 0.5
x_test = x_test.astype("float32") / 255 - 0.5

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# 2.define hyperparameter
input_img = Input(shape=(784,))
encoding_dim = 2

# 3.build autoencoder neural network
# define encoder layers
encoded_1 = Dense(units=128, activation="relu")(input_img)
encoded_2 = Dense(units=64, activation="relu")(encoded_1)
encoded_3 = Dense(units=10, activation="relu")(encoded_2)
encoded_output = Dense(units=encoding_dim)(encoded_3)

# define decoder layers
decoded_1 = Dense(units=10, activation="relu")(encoded_output)
decoded_2 = Dense(units=64, activation="relu")(decoded_1)
decoded_3 = Dense(units=128, activation="relu")(decoded_2)
decoded_output = Dense(units=784, activation="tanh")(decoded_3)

# construct the autodecoder model and encoder model
autodecoder = Model(input=input_img, output=decoded_output)
encoder = Model(input=input_img, output=encoded_output)

# compile the autoencoder
autodecoder.compile(
    optimizer="adam",
    loss="mse"
)

# 4.training model
autodecoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True
                )

# 5.plotting
encoded_imgs = encoder.predict(x_test)  # encoder.shape = [10000, 2]
print(encoded_imgs.shape)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()
