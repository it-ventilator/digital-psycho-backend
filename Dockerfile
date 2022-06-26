FROM python:3.9
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
RUN apt update && apt install -y cmake ffmpeg libsm6 libxext6
RUN pip install -r requirements.txt
COPY . /code/
EXPOSE 8000