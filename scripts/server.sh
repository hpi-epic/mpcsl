#!/bin/bash

# scripts/server.sh: Launch the application and any extra required processes locally.

set -e

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-nginx.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi


echo "==> Update…"
bash scripts/update.sh

echo "==> Run application…"
if [ -n "$1" ]; then
  # pass arguments to call. This is useful for starting in detached mode.
   docker-compose ${COMPOSE_FILE} up "$1"
else
   docker-compose ${COMPOSE_FILE} up
fi
