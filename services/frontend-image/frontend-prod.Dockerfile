FROM node:latest as build-stage

WORKDIR /app
COPY ./src/package.json /app/
COPY ./src/yarn.lock /app/

RUN yarn install

COPY ./src/ .

RUN yarn build

FROM nginx:latest

COPY --from=build-stage /app/build/ /usr/share/nginx/html

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

COPY ./swagger/ /usr/share/nginx/swagger
