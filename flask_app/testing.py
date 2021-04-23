#!/usr/bin/env python3

import json
import boto3

client = boto3.client('lambda')
s3 = boto3.client('s3')

BUCKET = 'music-genre-input'
TESTFILE = 'file_example_MP3_700KB.mp3'
LAMBDA_NAME = 's3-lambda'
print(s3.get_object(Bucket = BUCKET, Key=TESTFILE))

event = {
    "bucket": BUCKET,
    "key": TESTFILE
}

#print(json.dumps(event))

result = client.invoke(FunctionName=LAMBDA_NAME, InvocationType='RequestResponse', Payload=json.dumps(event))
range = result['Payload'].read()
response = json.loads(range)
print(response)
