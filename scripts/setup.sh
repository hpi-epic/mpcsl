#!/bin/bash

# scripts/setup.sh: Set up application for the first time after cloning, or set it
#                   back to the initial first unused state.

set -e

# ensure configurations are up to date
if [[ ! -f conf/backend.env ]]; then
    cp confdefault/backend.env conf/backend.env
fi
if [[ ! -f conf/algorithms.json ]]; then
    cp confdefault/algorithms.json conf/algorithms.json
fi


echo "==> Resetting everything (including database)…"
docker-compose -f docker-compose.yml --log-level ERROR down
docker-compose -f docker-compose-staging.yml --log-level ERROR down
docker-compose -f docker-compose-prod.yml --log-level ERROR down

echo "==> Update…"
bash scripts/update.sh
