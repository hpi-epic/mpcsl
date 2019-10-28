#!/bin/bash

# scripts/server.sh: Launch the application and any extra required processes locally.

set -e

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-prod.yml'
elif [[ "${MPCI_ENVIRONMENT}" = "staging" ]]; then
    COMPOSE_FILE='-f docker-compose-staging.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi


echo "==> Update…"
bash scripts/update.sh

echo "==> Run application…"
# pass arguments to call. This is useful for starting in detached mode.
docker-compose --project-name mpci --project-name mpci ${COMPOSE_FILE} up "$@"
