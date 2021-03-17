import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv3D, MaxPool3D, UpSampling3D
from keras import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from file_processing import load_files_list

path = '../Datasets/s_a_d__datasets/3D_descripted/'
folders_list = load_files_list(path)
X = []

for folder in folders_list:
    sub_folder_list = load_files_list(path + folder)
    for sub_folder in sub_folder_list:
        file_list = load_files_list(path + folder + '/' + sub_folder)
        for audio in file_list:
            X.append(np.load(path + folder + '/' + sub_folder + '/' + audio))


X = np.asarray(X)

X_train, X_test = train_test_split(X, test_size=0.3, random_state=5)

X_train = X_train.reshape(X_train.shape[0], 1, 16, 27, 128)
X_test = X_test.reshape(X_test.shape[0], 1, 16, 27, 128)

autoencoder = Sequential()

autoencoder.add(Conv3D(16, (3,3,3), activation='relu', padding='same', input_shape=(16,27,128,1)))
autoencoder.add(MaxPool3D(pool_size=(2,2,2)))
autoencoder.add(Conv3D(8, (3,3,3), activation='relu', padding='same'))
autoencoder.add(MaxPool3D(pool_size=(2,2,2)))

autoencoder.add(Conv3D(8, (3,3,3), activation='relu', padding='same'))

autoencoder.add(UpSampling3D(size=2))
autoencoder.add(Conv3D(8, (3,3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling3D(size=2))
autoencoder.add(Conv3D(16,(3,3,3), activation='relu', padding='same'))

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='accuracy', patience=5, verbose=1, mode='auto')]

history = autoencoder.fit(X_train, X_train, batch_size=10, epochs=50, callbacks=callbacks, validation_split=0.15)
