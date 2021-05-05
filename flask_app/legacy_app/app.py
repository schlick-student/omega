import boto3
import json
import os
from flask import Flask, render_template, request, redirect, send_file, url_for

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

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
BUCKET = 'music-genre-input'

@app.route('/')
def entry_point():
    contents = []
    return render_template('index.html', contents=contents)

@app.route("/upload", methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(UPLOAD_FOLDER, f.filename))
        upload_file(f"uploads/{f.filename}", BUCKET)

        return redirect(url_for('upload_success', filename=f.filename))

@app.route('/upload_success/<filename>')
def upload_success(filename):
    the_file = filename
    lambdaresponse = invokeLambda(filename)
    
    return render_template('index.html', fname=the_file, awslambda=lambdaresponse)

if __name__ == "__main__":
    application.run()
