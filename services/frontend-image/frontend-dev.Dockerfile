FROM node:latest as build-stage

WORKDIR /app
COPY ./static/ui/package.json /app/
COPY ./static/ui/yarn.lock /app/

RUN yarn install

COPY ./static/ui/ .

CMD [ "yarn", "start" ]