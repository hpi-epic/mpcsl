FROM python:3.7

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY job_scheduler.py seed.py server.py setup_algorithms.py setup.cfg ./
COPY confdefault ./confdefault
COPY conf ./conf
COPY scripts ./scripts
COPY migrations ./migrations
COPY src ./src
COPY static/swagger ./static/swagger
COPY test ./test
EXPOSE 5000
ENV MPCI_ENVIRONMENT production
CMD ["python", "server.py"]
