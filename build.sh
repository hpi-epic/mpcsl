#!/usr/bin/env bash
cd src/executionenvironments/r
docker build -t mpci_execution_r .
cd ../../..
docker-compose build