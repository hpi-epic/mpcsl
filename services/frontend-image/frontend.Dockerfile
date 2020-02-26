FROM nginx:latest

COPY ./h5bp/ /etc/nginx/h5bp

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

COPY ./swagger/ /usr/share/nginx/swagger

COPY ./build/ /usr/share/nginx/html