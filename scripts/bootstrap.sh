#!/bin/bash

# scripts/bootstrap.sh: Resolve all dependencies that the application requires to run

set -e

# ensure configurations are up to date
if [[ ! -f conf/backend.env ]]; then
    cp confdefault/backend.env conf/backend.env
fi
if [[ ! -f conf/algorithms.json ]]; then
    cp confdefault/algorithms.json conf/algorithms.json
fi

if [[ $(diff confdefault/backend.env conf/backend.env) ]]; then
    echo "==> Consider updating your conf/backend.env file…"
    diff confdefault/backend.env conf/backend.env || true
fi
if [[ $(diff confdefault/algorithms.json conf/algorithms.json) ]]; then
    echo "==> Consider updating your conf/algorithms.json file…"
    diff confdefault/algorithms.json conf/algorithms.json || true
fi

source conf/backend.env
if [[ "${MPCI_ENVIRONMENT}" = "production" ]]; then
    COMPOSE_FILE='-f docker-compose-nginx.yml'
    echo "==> Pull and build everything necessary (including UI)…"
    git submodule update --remote --init
else
    COMPOSE_FILE='-f docker-compose.yml'
    echo "==> Pull and build everything necessary (backend only)…"
fi

docker-compose ${COMPOSE_FILE} build --pull
