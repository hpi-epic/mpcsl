#!/bin/bash

# scripts/test.sh: Run test suite for application. Optionally pass in a path to an
#                  individual test file to run a single test.

EXIT_STATUS=0 # Execute all commands but fail at the end if one command failed

echo "==> Running flake8…"
docker-compose --project-name mpci run --rm backend flake8 || EXIT_STATUS=$?

echo "==> Running pytest…"
# pass arguments to test call. This is useful for calling a single test.
docker-compose --project-name mpci run --rm backend pytest --cov-report xml --cov=src "$@" || EXIT_STATUS=$?

exit $EXIT_STATUS
