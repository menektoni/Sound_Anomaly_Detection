import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Dropout

from functools import partial
import re
import os
import time

from file_processing import load_files_list
from file_processing import display_melspectrogram

files_list_csv = load_files_list('../Datasets/s_a_d__datasets/melspectrograms_esc50')

df = pd.read_csv('../Datasets/s_a_d__datasets/esc-50_tags.csv')

# We build a labels list and a targets list.
Images = []
Images_1D = []
Labels = []
i = 0
for file in files_list_csv:
    image_to_be = pd.read_csv('../Datasets/s_a_d__datasets/melspectrograms_esc50/' + file)
    Images_1D.append(image_to_be.to_numpy())
    Images.append(np.stack((Images_1D[i],)*3, axis=-1))
    Labels.append(re.split(r'-|\.', file)[3])
    i=+1

# We turn the lists into numpy arrays.
Images = np.asarray(Images)
Labels = np.asarray(Labels)
Images_1D = np.asarray(Images_1D)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Images, Labels, test_size=0.3, stratify=Labels)

y_train = pd.get_dummies(y_train)

from keras.applications import VGG16

model = VGG16(
    include_top=False, weights='imagenet', input_shape=[128, 217, 3])

model.trainable = False

vgg = Sequential()
for layer in model.layers:
    vgg.add(layer)

from keras.layers import GlobalAveragePooling2D

vgg.add(GlobalAveragePooling2D())
vgg.add(Dense(units=256, activation='relu')) # Each output has the 'Units' number of neurons
vgg.add(Dropout(0.5))
vgg.add(Dense(units=128, activation='relu'))
vgg.add(Dropout(0.5))
vgg.add(Dense(units=50, activation='softmax'))

vgg.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(X_train.shape, y_train.shape)
vgg.fit(X_train, y_train, epochs=100, batch_size=20)
