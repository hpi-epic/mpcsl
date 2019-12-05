FROM node:latest as build-stage

WORKDIR /app
COPY src/package.json /app/
COPY src/yarn.lock /app/

RUN yarn install

COPY src/ .

CMD [ "yarn", "start" ]