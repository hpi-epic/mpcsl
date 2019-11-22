FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY job_scheduler.py setup.cfg ./
COPY confdefault ./confdefault
COPY conf ./conf
COPY src ./src

EXPOSE 8080

CMD ["python", "job_scheduler.py"]
