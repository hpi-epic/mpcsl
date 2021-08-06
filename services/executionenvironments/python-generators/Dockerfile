FROM python:3.7

WORKDIR /app

COPY manm-cs/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY upload_requirements.txt .
RUN pip install -r upload_requirements.txt

COPY manm-cs/src src
COPY generate_and_upload.sh .
COPY upload_results.py .

ENV PYTHONPATH=.

ENTRYPOINT ["/bin/bash"]
