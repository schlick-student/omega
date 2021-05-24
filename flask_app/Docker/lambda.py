import json
import urllib.parse
import boto3
import librosa
import librosa.display
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image



#print('Loading function')

s3 = boto3.client('s3')

GENRE_LIST = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
                      'Instrumental', 'International', 'Pop', 'Rock']

"""
Runs the entire prediction flow when a new song is
uploaded.
:param filepath:
:return:
"""
def run_new_song_prediction(filepath, seconds_to_split, model_filepath, genre_list):

	split_mp3_file(filepath, seconds_to_split)
	time.sleep(30) # takes awhile for split songs to appear
	num_melspecs = create_melspecs_from_audio_clips()
	time.sleep(30) # takes awhile for melspecs to appear
	model = tf.keras.models.load_model(model_filepath)
	probabilities_list = predict_probabilities(model)
	final_probabilities_list = get_overall_predictions(num_melspecs,probabilities_list,genre_list)
	final_probabilities_list = sort_probabilities(final_probabilities_list)

	return final_probabilities_list

"""
Splits mp3 file into specified segment of seconds.
:param filename:
:param seconds_to_split:
:return:
 """
def split_mp3_file(filename, seconds_to_split):
	
	split_song = AudioSegment.from_file(filename, format="mp3")
	split = 0
	start = 0
	end = seconds_to_split * (split +1) *1000
	while(end < len(split_song)):
		splote = split_song[start:end]
		splote.export(str(split) + '-' + filename, format='mp3')
		split += 1
		start = seconds_to_split * (split) * 1000
		end = seconds_to_split * (split + 1) * 1000

"""
Takes split audio clips from memory and
converts them to melspectrograms
:return number_melspecs:
"""
def create_melspecs_from_audio_clips(filename):

	number_melspecs = 0 # keep track of number of melspecs
	files = os.listdir('.')
	for file in files:
		if(file.find(filename) > 0):
			testMelGenerator(file, file[:-4]+'.png')
			number_melspecs +=1
	return number_melspecs		

"""
Used to create melspectrograms
:param inpath:
:param outpath:
:return:
 """
def testMelGenerator(inpath, outpath):
	y,sr = librosa.load(inpath)
	plt.axis('off')
	plt.axis([0., 0., 1., 1.])
	S = librosa.feature.melspectrogram(y=y, sr=sr)
	librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
	plt.savefig(outpath, bbox_inches=None, pad_inches=0)
	plt.close()

def predict_probabilities(model, filename):
	
	prediction_list = []
	files = os.listdir('.')
	for file in files:
		if(file.find(filename) > 0):
			img = image.load_img(file, target_size=(160,240), color_mode='rgba')
			img_array = np.array(img).astype('float32') / 255
			img_batch = np.expand_dims(img_array, axis=0)
			prediction_list.append(model.predict(img_batch))
	return prediction_list

def get_overall_predictions(prediction_list, genre_list):
	all_probabilities_arr = np.array(prediction_list)
	all_probabilities_sum = all_probabilities_arr.sum(axis=0)
	final_probabilities = all_probabilities_sum / len(all_probabilities_arr)
	final_probabilities_list = final_probabilities.tolist()
	final_probabilities_list = [item for sublist in final_probabilities_list for item in sublist]
	final_probabilities_list = [list(a) for a in zip(final_probabilities_list, genre_list)]
	
	return sorted(final_probabilities_list, reverse=True)
	
def jsonify_response(final_probabilities):
	response = {
		'genre1' : final_probabilities[0][1],
		'prob1' : final_probabilities[0][0],
		'genre2' : final_probabilities[1][1],
		'prob2' : final_probabilities[1][0],
		'genre3' : final_probabilities[2][1],
		'prob3' : final_probabilities[2][0]
		}
	return response

def delete_files(filename):
	files = os.listdir('.')
	for file in files:
		if(files.find(filename) > 0):
			os.remove(file)
		elif(files.find(filename[:-4]+'.png') > 0):
			os.remove(file)	

split_mp3_file("testing.mp3", 3)
create_melspecs_from_audio_clips("testing.mp3")
create_melspecs_from_audio_clips("testing.mp3")
model = tf.keras.models.load_model("modelfit1")
preds = predict_probabilities(model, "testing.png")
final_preds = get_overall_predictions(preds, GENRE_LIST)
print(jsonify_response(final_preds))
delete_files("testing.mp3")

'''
def lambda_handler(event, context):
	bucket = event['bucket']
	key = event['key']
	print(bucket)
	print(key)
	print("Received event: " + json.dumps(event, indent=2))
	try:
		s3.download_file(bucket, key, '/tmp/test.mp3')
		y, sr = librosa.load('/tmp/test.mp3')
		#y, sr = librosa.load(librosa.ex('nutcracker'))
		tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
		response = {
			'beats' : ('Estimated tempo: {:.2f} beats per minute'.format(tempo))
			}
		print("Success")
		return response['beats']
	except Exception as e:
		print(e)
		print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
		raise e
'''
