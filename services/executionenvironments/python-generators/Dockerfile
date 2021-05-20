FROM python:3.7

WORKDIR /app

COPY manm-cs/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY manm-cs/src src
ENV PYTHONPATH=.

ENTRYPOINT ["python"]
