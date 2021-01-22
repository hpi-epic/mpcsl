FROM python:3.7

WORKDIR /app

COPY mpci-dag/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY mpci-dag/src src
ENV PYTHONPATH=.

ENTRYPOINT ["python"]
