FROM chris89/mpci_r

COPY ./requirements.r ./requirements.r
RUN Rscript requirements.r

COPY ./pcalg /scripts/src
WORKDIR /scripts/src
RUN R CMD INSTALL ./

COPY . /scripts
WORKDIR /scripts
ENTRYPOINT ["Rscript"]
