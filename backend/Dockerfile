FROM python:3.9.17-slim-buster
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . /app

# Set the environment variable for Uvicorn
ENV UVICORN_PORT=8000
ENV UVICORN_HOST=0.0.0.0

# Run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
