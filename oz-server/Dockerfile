FROM python:3.7-slim-stretch
RUN apt-get update -q -y && apt-get install -q -y python3-dev gcc

WORKDIR /usr/src/app/oz-server
COPY requirements.txt .
COPY setup.py .
RUN pip3 install -e .

COPY . ./

ENV TFHUB_CACHE_DIR=/usr/src/app/oz-server/hub_modules
ENV PYTHONUNBUFFERED=0

RUN useradd -ms /bin/bash oz
RUN mkdir -p oz /home/oz
RUN chown oz /home/oz
# USER oz
EXPOSE 8080