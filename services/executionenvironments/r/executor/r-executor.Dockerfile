FROM chris89/mpci_r

COPY ./requirements.r ./requirements.r
RUN Rscript requirements.r

COPY ./pcalg /scripts/pcalg
WORKDIR /scripts/pcalg
RUN R CMD INSTALL ./

COPY ./bnlearn /scripts/bnlearn
WORKDIR /scripts/bnlearn
RUN R CMD INSTALL ./

COPY . /scripts
WORKDIR /scripts
ENTRYPOINT ["Rscript"]
