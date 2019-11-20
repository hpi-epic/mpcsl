FROM node:latest as build-stage

WORKDIR /app
COPY ./static/ui/package.json /app/
COPY ./static/ui/yarn.lock /app/

RUN yarn install

COPY ./static/ui/ .

RUN yarn build

FROM nginx:latest

COPY --from=build-stage /app/build/ /usr/share/nginx/html

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

COPY ./static/swagger /usr/share/nginx/swagger
