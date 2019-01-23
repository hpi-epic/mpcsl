FROM danthe1/mpci_backend:latest

RUN apt-get install -y libgsl-dev libarmadillo-dev cmake libboost-all-dev
COPY requirements.txt /app/
COPY requirements.r /app/
WORKDIR /app
RUN Rscript requirements.r
RUN pip install -r requirements.txt
COPY conf confdefault src static/swagger test deploy_checker.py nginx.conf requirements.r requirements.txt seed.py server.py setup.cfg uwsgi.ini ./
RUN mkdir /app/master/executor/algorithms/cpp/parallel-pc/build && cd /app/master/executor/algorithms/cpp/parallel-pc/build && cmake .. && make && cp ParallelPC.out /usr/local/bin/ && cd -
CMD ["python", "server.py"]
