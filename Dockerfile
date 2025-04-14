# Docker version 27.3.1
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg

WORKDIR /app

# CGSTVG requirements
COPY ./requirements.txt ./reqs.txt
RUN pip install --only-binary=:all: -r reqs.txt
# RUN pip install -r reqs.txt

# JEPA requirements
COPY ./JEPA/requirements.txt ./JEPA/requirements.txt
COPY ./JEPA/setup.py ./JEPA/setup.py
WORKDIR /app/JEPA
RUN pip install .

RUN pip list --format=freeze > /app/requirements_frozen.txt

WORKDIR /app

COPY ./ ./

ENV PYTHONPATH="/app:/app/JEPA:/app/JEPA/new"

ENV PYTHONWARNINGS="ignore"

ENV PYTHONUNBUFFERED="1"

CMD [ "bash", "vidstg.sh" ]
# CMD [ "bash", "JEPA/new/vjepa.sh" ]
