FROM nginx:latest

COPY ./src/build/ /usr/share/nginx/html

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

COPY ./swagger/ /usr/share/nginx/swagger
