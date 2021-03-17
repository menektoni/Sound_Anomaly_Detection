import numpy as np
import pandas as pd

import librosa

import librosa.display
import matplotlib.pyplot as plt

import os
import re
import argparse


# Load the audiofiles inside a directory.
def load_files_list(directory):
    return os.listdir(directory)

# Load the audio given a file_path and a paramters.
def load_audio(single_file):
    # Take in count that our dataset files lasts everyone of them the same time. If not you should do something.
    audio, sr = librosa.load(single_file, duration=5.0)
    return audio, sr

# Create the Mel Spectrogram with the parameters passed before.
def create_melspec(params, audio_data, sampling_rate):
    S = librosa.feature.melspectrogram(audio_data, sr=sampling_rate, n_mels=params['n_mels'],
                                      hop_length=params['hop_length'], n_fft=params['n_fft'],
                                      fmin=params['fmin'], fmax=(sampling_rate//2))
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = S_dB.astype(np.float32)

    return S_dB

# Displaying a concrete spectrogram:
def display_melspectrogram(mel):
    mel_image = librosa.display.specshow(mel, sr=sr, hop_length=512, x_axis='time',
                                         y_axis='mel')
    return mel_image # It can be useful to use the next line of code in order to have a legend for the intensity.
# plt.colorbar(format='%+2.0f dB')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='''The purpose of this script is to turn a sound into a Dataframe of it spectrogram.''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-ip', '--input_path', default='/home/toni_domenech/Python/Datasets/s_a_d__datasets/audio_esc50', type=str, help='''From where do you want to read the audio.''')
    parser.add_argument('-od', '--output_directory', default='/home/toni_domenech/Python/Datasets/s_a_d__datasets/melspectrograms_esc50/', type=str, help='Output folder for the csv.')


    args = parser.parse_args()

    path_directory = args.input_path



    files_list = load_files_list(args.input_path)



# Creation of a dataframe following the exclusive patterns of ESC-50.
# This have to be changed if your dataset doesn't follow the same structure than ours.
    my_dict = [{'file': file, 'take': re.split(r'-|\.', file)[2], 'target': re.split(r'-|\.', file)[3]} for file in files_list]
    df = pd.DataFrame(my_dict)
    df.to_csv('esc-50.csv')

# List with all the os paths in order to read the data. This is maleable depending on your dataset.
    paths_list = []

    for file in files_list:
        paths_list.append(args.input_path + '/' + file) # Moldejar




#The Mel Spectrogram params:
    melspec_params = {
        'n_mels': 128, # The entire frequency spectrum divided by a concrete number.
        'duration': 5*22050, # Number of seconds * sample rate
        'hop_length': 512, # It has something to do with the duration. I think it fills the space with repetitions
        'n_fft': 2048, # Length of the Fast Fourier Transformation
        'fmin': 20
}


# Loading of the melspecs into .csv and leave them in the same folder
    for file, path in zip(files_list, paths_list):
        audio_file, sr = load_audio(path)
        melspec = create_melspec(melspec_params, audio_file, sr)
        df = pd.DataFrame(melspec)
    # It remains how to load it all on the same folder.
        df.to_csv(args.output_directory + file[:-4]+'.csv')
