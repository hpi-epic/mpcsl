#!/bin/bash

# scripts/setup.sh: Set up application for the first time after cloning, or set it
#                   back to the initial first unused state.

set -e

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-nginx.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi


echo "==> Resetting everything (including database)…"
docker-compose ${COMPOSE_FILE} down

echo "==> Update…"
bash scripts/update.sh
