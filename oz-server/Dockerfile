FROM python:3.7-alpine3.9

WORKDIR /usr/src/app/oz-server
COPY . ./

RUN pip install -e .
ENV PYTHONUNBUFFERED=0

EXPOSE 8080