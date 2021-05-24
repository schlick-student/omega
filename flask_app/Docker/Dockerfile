FROM public.ecr.aws/lambda/python:3.8


	
RUN \
  yum install which wget tar gzip xz libsndfile libsndfile-devel -y && \
  pip install --upgrade pip && \
  pip install soundfile librosa scikit-image

# make and enter new temp directory; get the files from johnvansickle to convert
# mp3 to wav files and extract
RUN \
  mkdir /tmp/ffmpeg/; cd /tmp/ffmpeg/ && \
  pwd && \
  wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && \
  ls && \
  tar xvf ffmpeg-git-amd64-static.tar.xz && \
  mv ffmpeg-git-20210425-amd64-static/ffmpeg ffmpeg-git-20210425-amd64-static/ffprobe /usr/local/bin/ && \
  which ffmpeg
  
  
#ENV PATH /tmp/ffmpeg/ffmpeg-git-20210425-amd64-static/ffmpeg:${PATH}

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

COPY lambda.py ./

CMD ["lambda.lambda_handler"]
