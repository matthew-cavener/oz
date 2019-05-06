FROM python:3.7-alpine3.9

# # RUN apk add --update --no-cache pkgconfig freetype png
# RUN apk add --update curl gcc g++ \
#     && rm -rf /var/cache/apk/*

# RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

# RUN pip install --no-cache-dir numpy scipy pandas matplotlib

WORKDIR /usr/src/app/oz-server
COPY . ./

RUN pip install -e .

ENV PYTHONUNBUFFERED=0

EXPOSE 8080