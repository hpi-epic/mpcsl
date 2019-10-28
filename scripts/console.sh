#!/bin/bash

# script/console: Launch a console for the application. Optionally allow an
#                 container name to be passed in to let the script handle the
#                 specific requirements for connecting to a console for that
#                 environment.

set -e

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-prod.yml'
elif [[ "${MPCI_ENVIRONMENT}" = "staging" ]]; then
    COMPOSE_FILE='-f docker-compose-staging.yml'
else
    COMPOSE_FILE='-f docker-compose.yml'
fi

echo "==> Open console…"
# pass arguments to call. This is useful for connecting to a different container than `backend`.
if [[ -n "$1" ]]; then
    SERVICE_NAME="$1"
else
    SERVICE_NAME="backend"
fi

echo "==> Run application…"
docker-compose 
docker-compose --project-name mpci ${COMPOSE_FILE} up -d

docker exec -it ${SERVICE_NAME} bash
