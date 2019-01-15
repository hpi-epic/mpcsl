FROM danthe1/mpci_backend:latest

COPY requirements.txt /app/
COPY requirements.r /app/
WORKDIR /app
RUN Rscript requirements.r
RUN pip install -r requirements.txt
COPY conf confdefault src static/swagger test deploy_checker.py nginx.conf requirements.r requirements.txt seed.py server.py setup.cfg uwsgi.ini ./
CMD ["python", "server.py"]
