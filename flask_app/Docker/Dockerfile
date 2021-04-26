FROM python:3.7-buster

RUN \
  pip install --upgrade pip wheel setuptools && \
  pip install librosa

COPY . .

ENTRYPOINT ./music_image_features.py
