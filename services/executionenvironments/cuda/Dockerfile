FROM nvidia/cuda:10.1-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive
# Install rbase
RUN apt-get update
RUN apt-get install -y r-base r-base-dev

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