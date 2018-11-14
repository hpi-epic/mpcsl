FROM python:3.6.7
ADD src /code
ADD requirements.txt /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD ["python", "server.py"]