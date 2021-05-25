#!/usr/bin/env python3

import boto3
import json
import os
from flask import Flask, render_template, request, redirect, send_file, url_for
from werkzeug.utils import secure_filename


# Upload file to S3
def upload_file(local_dir, file_name, s3bucket):
    object_name = file_name
    s3_client = boto3.client('s3', region_name='us-east-2')
    response = s3_client.upload_file(os.path.join(local_dir,file_name), s3bucket, object_name)
    # os.remove(file_name)
    return response


def list_files(bucket):
    s3 = boto3.client('s3')
    contents = []
    for item in s3.list_objects(Bucket=bucket)['Contents']:
        contents.append(item)

    return contents


def invokeLambda(filename):
    client = boto3.client('lambda', region_name='us-east-2')
    BUCKET = 'music-genre-input'
    TESTFILE = filename
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


application = Flask(__name__)
# Content Validation
application.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10  # 1 MB is about 1 minute audio, 10 MB max
application.config['ALLOWED_EXTENSIONS'] = ['.mp3', '.wav']

UPLOAD_FOLDER = 'uploads'
BUCKET = 'music-genre-input'


@application.route('/')
def entry_point():
    contents = []
    return render_template('index.html', contents=contents)


@application.route("/error")
def error_message():
    # TODO Return specific error message based on error code
    return render_template('error.html', errorMsg='The file must be less than 10 MB and .mp3 or .wav')


@application.route("/about", methods=['GET'])
def get_about():
    return render_template('about.html')

@application.route("/upload", methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # Validate Extensions before invokeLambda
        if f.filename != "":
            file_ext = os.path.splitext(f.filename)[1]
            if file_ext not in application.config['ALLOWED_EXTENSIONS']:
                return redirect(url_for('error_message'))
        # Ensure secure_filename()
        # Source: https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
        filename = secure_filename(f.filename)

        # Create uploads folder if not present in path
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        f.save(os.path.join(UPLOAD_FOLDER, filename))
        upload_file(UPLOAD_FOLDER, filename, BUCKET)
        return redirect(url_for('upload_success', filename=filename))


@application.route('/upload_success/<filename>')
def upload_success(filename):
    the_file = filename
    lambdaresponse = invokeLambda(filename)
    return render_template('index.html', fname=the_file, awslambda=lambdaresponse)


if __name__ == '__main__':
    application.run(host="0.0.0.0")
