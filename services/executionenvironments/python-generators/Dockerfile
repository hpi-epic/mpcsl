FROM python:3.7

WORKDIR /app

COPY upload_requirements.txt .
RUN pip install -r upload_requirements.txt

COPY generate_and_upload.sh .
COPY upload_results.py .

ENV PYTHONPATH=.

ENTRYPOINT ["/bin/bash"]
