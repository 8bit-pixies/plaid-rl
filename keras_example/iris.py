"""
Iris dataset using keras
"""

import os

import numpy as np
from sklearn.datasets import load_iris

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras.losses
from keras.layers import Activation, Dense
from keras.models import Sequential

iris = load_iris()

model = Sequential()
model.add(Dense(3, input_shape=(4,)))
model.add(Activation("softmax"))
model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer="sgd",
    metrics=["accuracy"],
)
model.fit(iris.data, iris.target, epochs=100)
