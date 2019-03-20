#!/bin/bash

# scripts/cibuild.sh: Setup environment for CI to run tests. This is primarily
#                     designed to run on the continuous integration server.

docker-compose --version

echo "==> Setup project…"
bash scripts/setup.sh

echo "==> Running tests…"
bash scripts/test.sh
