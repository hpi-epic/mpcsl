FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY conf ./conf
COPY src ./src
COPY test ./test
COPY migrations ./migrations
COPY seed.py server.py migration.py setup_algorithms.py setup.cfg ./

EXPOSE 5000
CMD ["python", "server.py"]
