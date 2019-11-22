#!/bin/bash

set -e

echo "==> Pushing Docker Images"

docker push milanpro/mpci_backend
docker push milanpro/mpci_frontend
docker push milanpro/mpci_scheduler
docker push milanpro/mpci_executor
