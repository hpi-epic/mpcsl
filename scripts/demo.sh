#!/bin/bash

# scripts/demo.sh: Start application with example experiment pre-configured

set -e

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-nginx.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi

echo "==> Setup…"
bash scripts/setup.sh

echo "==> Seed the database with example experiment"
docker-compose ${COMPOSE_FILE} run --rm backend python seed.py

echo "==> Run application…"
if [ -n "$1" ]; then
  # pass arguments to call. This is useful for starting in detached mode.
   docker-compose ${COMPOSE_FILE} up "$1"
else
   docker-compose ${COMPOSE_FILE} up
fi
