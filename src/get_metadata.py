# Team Name: Omega
# Last Updated: April 24, 2021
# Project: Top-n Music Genre Classification
# Program: Get Metadata
# Description: Extracts relevant metadata from mp3 files in the path: title, genre, duration. Saves in csv file
#              with audio track name. Then, creates a dataframe with these text features.

"""
Sources Cited
Title: What python library will provide efficient access to .mp3 metadata?
Source: https://github.com/quodlibet/mutagen/blob/master/mutagen/id3/_id3v1.py
Use: I used this library to extract relevant metadata from .mp3 files

Title: How do I resolve the error: charmap codec can't encode character?
Source: https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters
Use: I used information from this SO to resolve an error do to some metadata text entries using different alphabets.

Title: How can I constrain the function to only files that end in the appropriate extension?
Source: https://stackoverflow.com/questions/541390/extracting-extension-from-filename-in-python
Use: I used information from this SO to resolve an error associated with multiple file types in the directory.

Title: How can I iterate through multiple subdirectories efficiently?
Source: https://www.tutorialspoint.com/python/os_walk.htm
Use: I reviewed these tutorials to iterate through a directory with multiple sub-directories containing target files.

Title: How can I create a dataframe using a csv file?
Source: https://www.geeksforgeeks.org/creating-a-dataframe-using-csv-files/
Use: I used information from this article to create a dataframe from the cvs file of metadata.

"""
"""
Running Instructions

Run the code in the command line as follows:

    python3 get_metadata.py [path to dataset]
    
    If you do not specify a path, the script will prompt for path.
    
    Note: relative and absolute paths should both work. 
"""

# packages
from mutagen.mp3 import MP3
import os
import csv
import pandas
import sys


def get_metadata(filepath, file, filename):
    """
    Takes in the path to the audiofile, the name of the audiofile, and the
    name of the csv file the data will be written to.
    :param filepath: path to audiofile
    :param file: audiofile name
    :param filename: csv file name
    :return: none
    """
    audio = MP3(filepath)
    duration = audio.info.length
    if 'TCON' in audio:
        genre = audio['TCON'].text[0]
    else:
        genre = None
    if 'TIT2' in audio:
        title = audio['TIT2'].text[0]
    else:
        title = None
    trackID = file
    values = [trackID, title, genre, duration]
    with open(filename, "a", newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(values)


if __name__ == '__main__':
    # Open file and add field names
    fields = ['trackID', 'title', 'genre', 'duration']
    csvname = "metadata.csv"
    with open(csvname, "w", newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

    # get path to files
    if len(sys.argv) > 1:
        fmapath = sys.argv[1]
    else:
        fmapath = input('Enter path to audio files: ')

    # iterate through files in directory and get metadata
    for root, dirs, files in os.walk(fmapath):
        for file in files:
            filepath = os.path.join(root, file)
            prefix, extension = os.path.splitext(filepath)
            if extension == '.mp3':
                get_metadata(filepath, file, csvname)


    # make dataframe from file
    df = pandas.read_csv(csvname)
    print(df.head())
