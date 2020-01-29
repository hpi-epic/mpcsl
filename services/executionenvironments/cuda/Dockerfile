FROM nvidia/cuda:10.2-devel-ubuntu16.04
ENV DEBIAN_FRONTEND noninteractive
# Install rbase
RUN apt-get update
RUN apt-get install -y software-properties-common apt-transport-https ca-certificates
RUN apt-get update
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
RUN apt-get update
RUN apt-get install -y r-base

# Install r depedencies
RUN apt-get install -y libv8-dev libcurl4-openssl-dev libssl-dev
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile

COPY ./requirements.r ./requirements.r
RUN Rscript requirements.r

# Build
COPY ./cupc /scripts
WORKDIR /scripts
RUN nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu

# Copy other files
COPY . /scripts
WORKDIR /scripts

# Execute r-script
ENTRYPOINT ["Rscript"]