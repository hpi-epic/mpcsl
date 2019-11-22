FROM chris89/mpci_r

COPY ./src/executionenvironments/r/requirements.r ./requirements.r
RUN Rscript requirements.r

COPY ./src/executionenvironments/r/pcalg /scripts/src
WORKDIR /scripts/src
RUN R CMD INSTALL ./

COPY ./src/executionenvironments/r/ /scripts
WORKDIR /scripts
ENTRYPOINT ["Rscript"]
