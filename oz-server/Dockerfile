FROM tensorflow/tensorflow:latest-py3
# RUN apk add --update --no-cache pkgconfig freetype png

WORKDIR /usr/src/app/oz-server
COPY requirements.txt .
COPY setup.py .
RUN pip install -e .

COPY . ./

ENV TFHUB_CACHE_DIR=/usr/src/app/oz-server/hub_modules
ENV PYTHONUNBUFFERED=0

RUN useradd -ms /bin/bash oz
USER oz
EXPOSE 8080