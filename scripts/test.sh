#!/bin/bash

# scripts/test.sh: Run test suite for application. Optionally pass in a path to an
#                  individual test file to run a single test.

EXIT_STATUS=0 # Execute all commands but fail at the end if one command failed

echo "==> Running flake8…"
docker-compose --project-name mpci run --rm backend flake8 test/ src/ seed.py server.py setup_algorithms.py job_scheduler.py || EXIT_STATUS=$?

echo "==> Running pytest…"
docker-compose --project-name mpci run --rm backend pytest --cov-report xml --cov=src test/ || EXIT_STATUS=$?

exit $EXIT_STATUS
