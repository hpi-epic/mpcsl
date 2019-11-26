FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY confdefault ./confdefault
COPY conf ./conf
COPY scripts ./scripts
COPY migrations ./migrations
COPY static/swagger ./static/swagger
COPY test ./test
COPY src ./src
COPY job_scheduler.py seed.py server.py migration.py setup_algorithms.py setup.cfg ./

EXPOSE 5000
CMD ["python", "server.py"]
