import numpy as np
import cv2

from dataset_factory import DatasetFactory
from model_factory import Model
from constants import *

import tensorflow as tf
from tensorflow import keras

data_fact = DatasetFactory(train_path=TRAIN_PATH, test_path=TEST_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
model_fact = Model(TRAIN)

model = model_fact.get_model()

if LOAD:
    model.load_weights(SAVE_PATH)
    print(" - Modello caricato. - ")

else:
    train_data, val_data = data_fact.create_pandas_train_splitted_dataset()
    train_images, val_images = data_fact.create_images_train_splitted_dataset(train_data=train_data, val_data=val_data)

    history = model.fit(train_images,
                        epochs=10,
                        validation_data=val_images,
                        validation_steps=val_data.shape[0]//BATCH_SIZE,
                        steps_per_epoch=train_data.shape[0]//BATCH_SIZE
                        )
    if SAVE:
        model.save_weights(SAVE_PATH)
    print(history)

if TEST:
    test_data = data_fact.create_pandas_test_dataset()
    test_images = data_fact.create_images_test_dataset(test_data=test_data)

    pred = np.argmax(model.predict(test_images), axis=1)
    print(pred)

if SINGLE:
    # img = np.expand_dims(img, axis=0) bisogna anche modificare le dimensioni ecc ecc. crare un generatore che si occupa di questo cosa con batch_size = 1

    image_path = 'prova/1.jpeg'
    single_imgage = data_fact.create_images_single_dataset(image_path)
    pred = model.predict(single_imgage)
    print(pred)
    pred = np.argmax(pred, axis=1)
    print("It's a cat") if pred == 0 else print("It's a dog")
