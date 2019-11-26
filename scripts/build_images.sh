#!/bin/bash

set -e

echo "==> Building Docker Images"

docker build -t milanpro/mpci_backend -f docker/backend.Dockerfile .
docker build -t milanpro/mpci_frontend -f docker/frontend.Dockerfile .
docker build -t milanpro/mpci_scheduler -f docker/scheduler.Dockerfile .
docker build -t milanpro/mpci_executor -f docker/r-executor.Dockerfile ./src/executionenvironments/r/
