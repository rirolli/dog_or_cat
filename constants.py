import os

# PATH
TRAIN_PATH = 'train'
TEST_PATH = 'test1'
TRAIN_LIST = os.listdir("train")
TEST_LIST = os.listdir("test1")
SAVE_PATH = 'saves/model.h5'

# IMAGES
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

# TRAIN
TRAIN = False
LOAD = True and not TRAIN
TEST = False
SINGLE = True
SAVE = True
BATCH_SIZE = 15