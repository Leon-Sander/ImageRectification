FROM ubuntu:18.04
COPY . usr/app/
EXPOSE 8000
WORKDIR /usr/app/
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r docker_requirements.txt
CMD python3 server_folder/server.py