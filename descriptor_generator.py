import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D

from file_processing import melspec_params, load_files_list, load_audio, create_melspec
import argparse

def descriptor_encoders(model):
    new_model = keras.models.load_model(model)
    new_model.trainable = False

    descriptor_encoder_1D = keras.models.Sequential()
    descriptor_encoder_3D = keras.models.Sequential()

    for i in range(10):
        descriptor_encoder_1D.add(new_model.layers[i])
        if i<=8:
            descriptor_encoder_3D.add(new_model.layers[i])
    return descriptor_encoder_1D, descriptor_encoder_3D

melspec_params = {
    'n_mels': 128, # The entire frequency spectrum divided by a concrete number.
    'duration': 5*22050, # Number of seconds * sample rate
    'hop_length': 512, # It has something to do with the duration. I think it fills the space with repetitions
    'n_fft': 2048, # Length of the Fast Fourier Transformation
    'fmin': 20
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''The purpose of this script is to encode a sound.''',
     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ip', '--input_path', default='/home/toni_domenech/Python/Datasets/s_a_d__datasets/MiMii/normal_train/',
     type=str, help='''From where do you want to read the audio.''')
    parser.add_argument('-m', '--model', default='model_128_216_acc70.h5',
     type=str, help='''From where do you want to read the audio.''')
    parser.add_argument('-od1', '--output_directory_1D', default='/home/toni_domenech/Python/Datasets/s_a_d__datasets/abnormal_1D/',
     type=str, help='Output folder for the 1D descriptor.')
    parser.add_argument('-od3', '--output_directory_3D', default='/home/toni_domenech/Python/Datasets/s_a_d__datasets/abnormal_3D/',
     type=str, help='Output folder for the 1D descriptor.')


    args = parser.parse_args()

    descriptor_encoder_1D, descriptor_encoder_3D = descriptor_encoders(args.model)

    folders_list = load_files_list(args.input_path)
    for folder in folders_list:
        sub_folder_list = load_files_list(args.input_path + folder)
        for sub_folder in sub_folder_list:
            files_list = load_files_list(args.input_path + folder + '/' + sub_folder)
            for file in files_list:
                audio_file, sr = load_audio(args.input_path + folder + '/' + sub_folder + '/' + file)
                melspec = create_melspec(melspec_params, audio_file, sr)
                melspec = melspec.reshape(1, 128, 216, 1)
                encoded_melspec_1D = descriptor_encoder_1D.predict(melspec)
                encoded_melspec_3D = descriptor_encoder_3D.predict(melspec)


                np.save(args.output_directory_1D + folder + '/abnormal_' + sub_folder + '/' + file[:-4], encoded_melspec_1D)
                np.save(args.output_directory_3D + folder + '/abnormal_' + sub_folder + '/' + file[:-4], encoded_melspec_3D)
