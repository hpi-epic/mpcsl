FROM python:3.7.5

WORKDIR app
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["pytest", "--cov-report", "xml", "--cov=src", "test/"]
