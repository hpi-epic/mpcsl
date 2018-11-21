FROM python:3.6.7
COPY requirements.txt ./app/
WORKDIR ./app
RUN apt-get update && apt-get install -y r-base && apt-get install -y libv8-dev
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN Rscript -e "source('http://www.bioconductor.org/biocLite.R'); biocLite('graph'); biocLite('RBGL')" && Rscript -e "install.packages('optparse')" && Rscript -e "install.packages('pcalg')" && Rscript -e "install.packages('httr')"
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "server.py"]
