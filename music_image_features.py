"""
Sources Cited

Title: How can I save a Librosa spectrogram plot as a specific sized image?
Source:
Use: I used this SO article to build out the way we are turning the Mel
Spectrogram data into an image for our future neural net. The Mel
Spectrogram feature in Librosa gives the user an output of all the different Hz
frequencies over time. From this numeric data, we can create an image of the Mel
Spectrogram frequencies, which will be used for our image classifier.

Title: Using display.specshow - Librosa
Source: https://librosa.org/doc/main/auto_examples/plot_display.html
Use: I used this page of the Librosa documentation to understand how to output
the Mel Spectrogram data into a Matplotlib Plot.

Title: Feature Extraction - Librosa
Source: https://librosa.org/doc/latest/feature.html
Use: I used the feature extraction documentation from Librosa in order to
figure out which features could be extracted from an mp3 file once it was
turned into an image. This is the main source for the image features dataset
which will be used in training the neural net.

Title: Audio Data Analysis Deep Learning - Python Part 2
Source: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-2.html
Use: To better understand the type of data (aka png) that a CNN will take
into it in order to be trained.

"""

"""
Running Instructions

Run the code in the command line as follows:

 python3 music_image_features.py ./fma_large_data/fma_large/000/000002.mp3 track2.png
 
 If you don't give an output filename for the image png, then the script will
 give it a default name of 'music_image.png'.
 
 Note: Ignore the warning about PySoundFile failing. It's some issue with 
 Librosa and doesn't mean anything for us.
 
 Note: you can also change the num_bins and hop_length. I was just using 
 values used in examples I saw.
 
"""


import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import skimage.io
import sys


def audio_file_to_mel_spectrogram(filepath, num_bins, hop_length):
    """
    Takes in the audiofile and converts it to a Mel Spectrogram
    :param filepath [string]:

    :return mel_spect [numpy.ndarray]:
    """
    # Read in audio file
    x, sr = librosa.load(filepath, sr=None, mono=True, duration=5)

    # Get image window (aka image length)
    window = create_image_window(x, hop_length)

    # fourier transform
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=hop_length))

    # Get Mel Spectrogram Features
    mel_spect = librosa.feature.melspectrogram(y=window, sr=sr,
                                               S=stft ** 2,
                                               n_fft=hop_length*2,
                                               n_mels=num_bins,
                                               hop_length=hop_length)
    # Convert to Db
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return mel_spect


def mel_spectrogram_to_plot(mel_spectrogram, output_name):
    """
    Shows each audio clip as a Mel Spectrogram using Matplotlib to plot this
    feature
    :param mel_spectrogram:
    :param output_name (png file name -- string):
    :return None:
    """
    img = librosa.display.specshow(mel_spectrogram, y_axis='mel', fmax=8000,
                                   x_axis='time')
    plt.axis('off')
    plt.savefig(output_name)
    plt.clf()


def create_image_window(y, hop_length):
    """
    Creates how wide the image is with respect to the audio clip
    :param y:
    :param hop_length:
    :return:
    """
    time_steps = 384  # number of time-steps. Width of image

    # extract a fixed length window
    start_sample = 0  # starting at beginning

    length_samples = time_steps * hop_length

    window = y[start_sample:start_sample + length_samples]

    return window


def image_diminsions(X, min=0.0, max=1.0):
    """
    Scales the Mel Spectrogram data so it fits within an 8-bit range
    :param X:
    :param min:
    :param max:
    :return:
    """
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def mel_spectrogram_to_image(mel_spectrogram_data, output):
    """
    Converts Mel Spectrogram data into png images
    :param mel_spectrogram_data:
    :param output:
    :return:
    """

    # min-max scale to fit inside 8-bit range
    img = image_diminsions(mel_spectrogram_data, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img             # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(output, img)


if __name__ == '__main__':

    # If there is more than 1 argument, then a file or filepath was given as
    # well as the output file name.
    if len(sys.argv) > 1:

        file = sys.argv[1]

        # Check for output file name. If none exists, give it a default name
        if len(sys.argv) == 2:
            output_png = 'music_image.png'

        else:

            output_png = sys.argv[2]

        spec = audio_file_to_mel_spectrogram(file, num_bins=128,
                                             hop_length=512)

        mel_spectrogram_to_plot(spec, output_png)

    # Otherwise, if an incorrect argument was input or no file/filepath,
    # then quit.
    else:
        print('Incorrect Argument(s) Provided. Quitting.')
        exit()
