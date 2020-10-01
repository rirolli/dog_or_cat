import tensorflow as tf
import numpy as tf

from tensorflow import keras
from constants import *


class Model():

    def init_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
        return model

    def get_model(self):
        return self.model

    def __init__(self, train):
        self.model = self.init_model()
        if train:
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

