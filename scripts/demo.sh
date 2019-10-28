#!/bin/bash

# scripts/demo.sh: Start application with example experiment pre-configured

set -e

echo "==> Setup…"
bash scripts/setup.sh

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-prod.yml'
elif [[ "${MPCI_ENVIRONMENT}" = "staging" ]]; then
    COMPOSE_FILE='-f docker-compose-staging.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi

echo "==> Seed the database with example experiment"
docker-compose --project-name mpci ${COMPOSE_FILE} run --rm backend python seed.py

echo "==> Run application…"
# pass arguments to call. This is useful for starting in detached mode.
docker-compose --project-name mpci ${COMPOSE_FILE} up "$@"
