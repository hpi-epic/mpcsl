FROM chris89/mpci_r

COPY ./requirements.r ./requirements.r
RUN Rscript requirements.r
COPY . /scripts
ENTRYPOINT ["Rscript"]
