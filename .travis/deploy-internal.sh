#!/usr/bin/env bash

ssh deploy@vm-mpws2018-proj.eaalab.hpi.uni-potsdam.de /bin/bash <<EOT
cd mpci
docker-compose down
docker-compose build
docker-compose up --detach
EOT