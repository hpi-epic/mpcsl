#!/bin/bash

# scripts/cibuild.sh: Setup environment for CI to run tests. This is primarily
#                     designed to run on the continuous integration server.

echo "==> Update…"
bash scripts/update.sh

echo "==> Running tests…"
bash scripts/test.sh
