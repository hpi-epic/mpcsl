#!/bin/bash

# scripts/update.sh: Update application to run for its current checkout

set -e

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-nginx.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi


echo "==> Bootstrap…"
bash scripts/bootstrap.sh

echo "==> Updating database…"
# run all database migrations to ensure everything is up to date
docker-compose ${COMPOSE_FILE} run --rm backend flask db upgrade
