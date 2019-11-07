FROM python:3.7

COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
COPY conf confdefault src static/swagger scripts test nginx.conf requirements.txt server.py setup.cfg ./
CMD ["python", "server.py"]
