import pandas as pd
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import scipy
import skimage
from skimage.transform import resize
#from keras import backend as K
#K.common.image_dim_ordering()
print(os.listdir("C:/input/chest_xray/chest_xray"))

print(os.listdir("C:/input/chest_xray/chest_xray/train/"))

TRAIN_DIR = "C:/input/chest_xray/chest_xray/train/"
TEST_DIR =  "C:/input/chest_xray/chest_xray/test/"
VAL_DIR = "C:/input/chest_xray/chest_xray/val/"


def get_label(Dir):
    for nextdir in os.listdir(Dir):
        if not nextdir.startswith('.'):
            if nextdir in ['NORMAL']:
                label = 0
            elif nextdir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
    return nextdir, label



def preprocessing_data(Dir):
    X = []
    y = []

    for nextdir in os.listdir(Dir):
        nextdir, label = get_label(Dir)
        temp = Dir + nextdir

        for image_filename in tqdm(os.listdir(temp)):
            path = os.path.join(temp + '/', image_filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = skimage.transform.resize(img, (150, 150, 3))
                img = np.asarray(img)
                X.append(img)
                y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

# X_train, y_train = preprocessing_data(TRAIN_DIR)


def get_data(Dir):
    X = []
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['NORMAL']:
                label = 0
            elif nextDir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2

            temp = Dir + nextDir

            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = skimage.transform.resize(img, (150, 150, 3))
                    # img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

X_train, y_train = get_data(TRAIN_DIR)
X_test , y_test = get_data(TEST_DIR)
X_val, y_val=get_data(VAL_DIR)

np.save("X_train",X_train)
np.save("y_train",y_train)
np.save("X_test",X_test)
np.save("y_test",y_test)
np.save("X_val",X_val)
np.save("y_val",y_val)

