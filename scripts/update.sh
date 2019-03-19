#!/bin/bash

# scripts/update.sh: Update application to run for its current checkout

set -e

echo "==> Bootstrap…"
bash scripts/bootstrap.sh

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-prod.yml'
elif [[ "${MPCI_ENVIRONMENT}" = "staging" ]]; then
    COMPOSE_FILE='-f docker-compose-staging.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi

echo "==> Updating database…"
# run all database migrations to ensure everything is up to date
docker-compose ${COMPOSE_FILE} run --rm backend flask db upgrade
# update all algorithms from conf/algorithms.json
docker-compose ${COMPOSE_FILE} run --rm backend python setup_algorithms.py