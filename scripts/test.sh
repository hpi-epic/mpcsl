#!/bin/bash

# scripts/test.sh: Run test suite for application. Optionally pass in a path to an
#                  individual test file to run a single test.


echo "==> Running flake8…"
docker-compose run --rm backend flake8

echo "==> Running pytest…"
# pass arguments to test call. This is useful for calling a single test.
docker-compose run --rm backend pytest --cov-report xml --cov=src "$@"
