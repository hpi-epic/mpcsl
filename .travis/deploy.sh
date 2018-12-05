#!/usr/bin/env bash

#if [[ "$TRAVIS_BRANCH" = "master" && $TRAVIS_PULL_REQUEST = "false" ]]; then
if [[ "$TRAVIS_BRANCH" = "master" ]]; then
    echo "Deploying the master"
    eval "$(ssh-agent -s)" # start the ssh agent
    chmod 600 ./deploy_key
    echo -e "Host deploy@mpci.epic-hpi.de\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
    ssh-add ./deploy_key
    git remote add deploy ssh://deploy@mpci.epic-hpi.de:/home/deploy/mpci
    git push deploy master
    ssh deploy@mpci.epic-hpi.de <<EOF
      cd mpci
      docker-compose down
      docker-compose build
      docker-compose up --detach
EOF
fi