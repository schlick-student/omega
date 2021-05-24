#!/usr/bin/env python3

import json
import boto3
import librosa

client = boto3.client('lambda')
s3 = boto3.client('s3')

BUCKET = 'music-genre-input'
TESTFILE = 'file_example_MP3_5MG.mp3'
LAMBDA_NAME = 'docker-lambda'
#print(s3.get_object(Bucket = BUCKET, Key=TESTFILE))

event = {
    "bucket": BUCKET,
    "key": TESTFILE
}
'''
print(json.dumps(event))
s3.download_file(BUCKET, TESTFILE, './uploads/test.mp3')
y, sr = librosa.load('./uploads/test.mp3')
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
'''

result = client.invoke(FunctionName=LAMBDA_NAME, InvocationType='RequestResponse', Payload=json.dumps(event))
range = result['Payload'].read()
response = json.loads(range)
print(response)

