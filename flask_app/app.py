#!/usr/bin/env python3

import os
from flask import Flask, render_template, request, redirect, send_file, url_for
from s3 import upload_file, list_files

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
    return render_template('index.html', fname=the_file)

if __name__ == '__main__':
    app.run(debug=True)
