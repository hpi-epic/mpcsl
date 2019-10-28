#!/bin/bash

# scripts/bootstrap.sh: Resolve all dependencies that the application requires to run

set -e

# build execution environments (separate docker images)
cd src/executionenvironments/r
docker build -t mpci_execution_r .
cd ../../..

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-prod.yml'
    echo "==> Pull and build everything necessary (including UI)…"
    git submodule update --remote --init
elif [[ "${MPCI_ENVIRONMENT}" = "staging" ]]; then
    COMPOSE_FILE='-f docker-compose-staging.yml'
    echo "==> Pull and build everything necessary (including UI)…"
    git submodule update --remote --init
else
    COMPOSE_FILE='-f docker-compose.yml'
    echo "==> Pull and build everything necessary (backend only)…"
fi

docker-compose --project-name mpci ${COMPOSE_FILE} build
