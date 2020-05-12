FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN cat ./requirements.txt | xargs -n 1 -L 1 pip install

COPY . /scripts
WORKDIR /scripts

ENTRYPOINT ["python"]
