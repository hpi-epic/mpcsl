FROM python:3.6.7
COPY . ./app
WORKDIR ./app
RUN pip install -r requirements.txt
CMD ["python", "server.py"]