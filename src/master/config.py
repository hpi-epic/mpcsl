import os

# App
APP_HOST = os.environ.get('APP_HOST', '0.0.0.0')
APP_PORT = os.environ.get('APP_PORT', '5000')

if APP_PORT != '80':
    SERVER_NAME = 'localhost:' + APP_PORT
else:
    SERVER_NAME = 'localhost'

# Database
DB_TYPE = os.environ.get('DB_TYPE', 'postgresql')
DB_HOST = os.environ.get('DB_HOST', 'database')
DB_DATABASE = os.environ.get('DB_DATABASE', 'postgres')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')

SQLALCHEMY_DATABASE_URI = os.environ.get(
    'SQLALCHEMY_DATABASE_URI',
    DB_TYPE + '://' + DB_USER + ':' + DB_PASSWORD + '@' +
    DB_HOST + '/' + DB_DATABASE
)

SQLALCHEMY_TRACK_MODIFICATIONS = os.environ.get(
    'SQLALCHEMY_TRACK_MODIFICATIONS',
    'false'
).lower() == 'true'
