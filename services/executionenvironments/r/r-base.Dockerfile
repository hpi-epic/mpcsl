FROM r-base

RUN apt-get update && apt-get install -y libv8-dev libcurl4-openssl-dev libssl-dev
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile

COPY ./requirements.r /scripts/requirements.r
WORKDIR ./scripts
RUN Rscript requirements.r
ENTRYPOINT ["Rscript"]
