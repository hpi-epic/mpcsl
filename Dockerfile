FROM danthe1/mpci_backend:latest

COPY requirements.txt ./app/
COPY requirements.r ./app/
WORKDIR ./app
RUN Rscript requirements.r
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "server.py"]
