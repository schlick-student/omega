import json
import urllib.parse
import boto3
import librosa

print('Loading function')

s3 = boto3.client('s3')

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
