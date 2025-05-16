FROM python:3.9-slim

LABEL Name="PIPER TTS SERVER"

LABEL Description="TTS Service for Piper(CPU inference) model"

# Added -y flag to automatically approve installation
RUN apt update && apt-get install -y ffmpeg

COPY requirements.txt requirements.txt
# Fixed path to requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app app

ENV APP_PORT=9620

# Changed to absolute path for WORKDIR
WORKDIR /app

ENTRYPOINT ["python3", "server.py"]
