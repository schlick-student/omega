"""
Sources Cited

Title: Iterating Through Directories with Python
Site: https://stackoverflow.com/questions/19587118/iterating-through-directories-with-python
Use:

Title: How can I Iterate Over Files in a Given Directory
Site: https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
Use: I used this as the basis of my code for iterating through all the mp3
files in fma_small. This allowed me to grab each file and perform the needed
methods on them in order to create my feature dataset.

Title: How to load, convert, and save imaegs with the keras api
Site: https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/
Use: I used this site for converting .png images into an array. Each array
is an image and is composed of its RGB colors at each pixel in the image.

Title: Audio Data Analysis Deep Learning - Python Part 2
Source: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-2.html
Use: To better understand the type of data (aka png) that a CNN will take
into it in order to be trained.

"""
import audioread
import music_image_features as m
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pandas as pd


def get_track_number(file):
    """
    Removes .mp3 from file in order to get track ID
    :param file:
    :return:
    """
    sep = '.'
    track_id = file.split(sep, 1)[0]

    return track_id


def filename_to_png(directory, file):
    sep = '.'
    png_name = file.split(sep, 1)[0]
    png_name = directory + png_name + '.png'

    return png_name


def load_image(image_name):
    """
    Loads image using Keras
    :param image_name [string]:
    :return img [object]:
    """
    img = load_img(image_name)
    return img


def image_data_to_array(image):
    """
    Converts image to an array (with RGB data) and outputs it to a numpy array
    :param image:
    :return img_array [np.array]:
    """
    img_array = img_to_array(image)

    return img_array


def mp3_file_to_pixel_array(directory_name):
    """
    Takes in the directory name to iterate through all the files.
    Returns a pandas dataframe of each image and it's pixel RGBs.
    :param directory_name (string):
    :return image_feature_df (pandas dataframe):
    """
    track_id_list = []

    # Numpy shape=(128, 385, 3) -- important for modeling later
    # length after clean-up (aka mp3 fliles left) is 7802

    image_feature_df = pd.DataFrame()

    # Now, iterate through all images, and save them to a pd dataframe
    for filename in os.listdir(directory_name)[6001:7802]:
        if filename.endswith(".png"):
            img = load_image(filename)
            img = image_data_to_array(img)

            # There are 3 tracks that are not the same size
            if img.shape[1] == 385:
                track_id = get_track_number(filename)
                track_id_list.append(str(track_id))

                # make array flat
                a = np.ravel(img)
                a = np.append(track_id, a)
                image_feature_df[filename] = a

    return image_feature_df


def save_df_data(dataframe):
    """
    Make data manipulations to the dataframe created from turing each image
    into a pixel by pixel RGB dataset
    :param dataframe:
    :return:
    """
    print('transposing df')
    image_feature_df = dataframe.T
    image_feature_df = dataframe.rename(columns={0: 'track_id'})
    print(image_feature_df.head())

    # Save as csv & pickle
    print('saving to csv')
    image_feature_df.to_csv('image_data_7.csv', index=False)
    image_feature_df.to_pickle("image_data.pkl")


def main():
    rootdir = './fma_small/'
    dir = os.path.dirname(os.path.realpath(__file__))

    for subdir, dirs, files in os.walk(rootdir):
        try:
            for file in files:
                #print(os.path.join(subdir, file))

                if file == '' or file == 'README.txt' or file == 'checksums' \
                        or os.path.join(subdir,
                                        file) == './fma_small/.DS_Store':
                    continue

                else:

                    # Get mel spectrogram data from mp3 files -- converts
                    # mp3 to mel spectrogram
                    spect = m.audio_file_to_mel_spectrogram(
                        os.path.join(subdir, file), num_bins=128,
                        hop_length=512)

                    # Save mel spectrogram to new directory for training
                    m.mel_spectrogram_to_plot(spect,filename_to_png(
                        './spectrographs/', file))

        # If error opening file, move on to next file. This is most likely
        # due to there being .mp3 files with 0 seconds of recording saved
        except audioread.exceptions.NoBackendError:
            pass


main()
