import boto3
import json

#Upload file to S3
def upload_file(file_name, s3bucket):
    object_name = file_name
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(file_name, s3bucket, object_name)

    return response

def list_files(bucket):
    s3 = boto3.client('s3')
    contents = []
    for item in s3.list_objects(Bucket=bucket)['Contents']:
        content.append(item)

    return contents
    
def invokeLambda(filename):
	client = boto3.client('lambda')
	BUCKET = 'music-genre-input'
	TESTFILE = 'uploads/' + filename
	print(TESTFILE)
	LAMBDA_NAME = 'docker-lambda'
	event = {
    		"bucket": BUCKET,
    		"key": TESTFILE
	}
	result = client.invoke(FunctionName=LAMBDA_NAME, InvocationType='RequestResponse', Payload=json.dumps(event))
	range = result['Payload'].read()
	response = json.loads(range)
	return response
