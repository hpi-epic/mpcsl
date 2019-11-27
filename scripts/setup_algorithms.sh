#!/bin/bash

# scripts/setup.sh: Set up application for the first time after cloning, or set it
#                   back to the initial first unused state.

set -e

# ensure configurations are up to date

if [[ ! -f conf/algorithms.json ]]; then
    cp confdefault/algorithms.json conf/algorithms.json
fi

echo "==> Update…"

# ensure configurations are up to date
if [[ $(diff confdefault/algorithms.json conf/algorithms.json) ]]; then
    echo "==> Consider updating your conf/algorithms.json file…"
    diff confdefault/algorithms.json conf/algorithms.json || true
fi

