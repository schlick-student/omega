FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install Flask
RUN pip install boto3
EXPOSE 5000
CMD ["python", "application.py"]
