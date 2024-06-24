FROM python:3.10

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY conf ./conf
COPY src ./src
COPY job_scheduler.py setup.cfg ./
EXPOSE 8080

CMD ["python", "job_scheduler.py"]
