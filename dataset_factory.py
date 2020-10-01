import pandas as pd
import tensorflow as tf
import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

class DatasetFactory:
    def __init__(self, train_path, test_path, image_size, batch_size=15):
        self.train_path = train_path
        self.test_path = test_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.image_train_generator, self.image_test_validation_generator = self.create_images_generators()

    def create_images_generators(self):
        train_generator = ImageDataGenerator(
            rotation_range=15,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        test_validation_generator = ImageDataGenerator(rescale=1./255)
        return train_generator, test_validation_generator

    # TRAIN
    def create_pandas_train_dataset(self, train_path=None):
        if train_path is None:
            train_path = self.train_path
        dog_cat_train = {}
        for filename in os.listdir(train_path):
            dog_cat_train.update([(filename, filename.split('.')[0])])
        data = pd.DataFrame(dog_cat_train.items(), columns=['filename', 'category'])
        # data = data.replace({'category' : {'cat': 0, 'dog': 1}})
        # print(data.head())
        # print(data.tail())
        return data

    def create_pandas_train_splitted_dataset(self, test_size=.2, train_path=None):
        if train_path is None:
            train_path = self.train_path
        data = self.create_pandas_train_dataset(train_path=train_path)
        train, validation = train_test_split(data, test_size=test_size)
        return train, validation

    def create_images_train_dataset(self, batch_size=None, data=None):
        if data is None:
            data = self.create_pandas_train_dataset()
        if batch_size is None:
            batch_size = self.batch_size
        train_generator = self.image_train_generator.flow_from_dataframe(
            data,
            self.train_path,
            x_col='filename',
            y_col='category',
            target_size=self.image_size,
            class_mode='categorical',
            batch_size=batch_size
        )
        return train_generator

    def create_images_validation_dataset(self, batch_size=None, data=None):
        if data is None:
            raise("Invalid input data.")
        if batch_size is None:
            batch_size = self.batch_size
        train_generator = self.image_test_validation_generator.flow_from_dataframe(
            data,
            self.train_path,
            x_col='filename',
            y_col='category',
            target_size=self.image_size,
            class_mode='categorical',
            batch_size=batch_size
        )
        return train_generator

    def create_images_train_splitted_dataset(self, train_data=None, val_data=None):
        if train_data is None or val_data is None:
            train_data, val_data = self.create_pandas_train_splitted_dataset()
        train_images_dataset = self.create_images_train_dataset(data=train_data)
        validation_images_dataset = self.create_images_validation_dataset(data=val_data)
        return train_images_dataset, validation_images_dataset

    # TEST
    def create_pandas_test_dataset(self, test_path=None):
        if test_path is None:
            test_path = self.test_path
        dog_cat_test = []
        for filename in os.listdir(test_path):
            dog_cat_test.append(filename)
        return pd.DataFrame(dog_cat_test, columns=['filename'])

    def create_images_test_dataset(self, test_data=None, batch_size=15):
        if test_data is None:
            test_data = self.create_pandas_test_dataset()
        if batch_size is None:
            batch_size = self.batch_size
        test_images_dataset = self.image_test_validation_generator.flow_from_dataframe(
            test_data,
            self.test_path,
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=self.image_size,
            batch_size=batch_size,
            shuffle=False
        )
        return test_images_dataset

    # SINGLE
    def create_images_single_dataset(self, path_name:str):
        if path_name is None:
            raise("Path name unknow")

        path_name_list = path_name.split('/')
        path_name_root = '/'.join(path_name_list[:-1])
        file_name = path_name_list[-1]

        single_data = pd.DataFrame([file_name], columns=['filename'])
        single_images_dataset = self.image_test_validation_generator.flow_from_dataframe(
            single_data,
            path_name_root,
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=self.image_size,
            batch_size=1,
            shuffle=False
        )
        return single_images_dataset
