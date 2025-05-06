# Docker version 27.3.1
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg

WORKDIR /app

# CGSTVG requirements
COPY ./requirements.txt ./reqs.txt
RUN pip install --only-binary=:all: -r reqs.txt

RUN pip list --format=freeze > /app/requirements_frozen.txt

COPY ./ ./

ENV PYTHONPATH="/app:"

CMD [ "bash", "vidstg.sh" ]
