import os

# App
APP_HOST = os.environ.get('APP_HOST', '0.0.0.0')
APP_PORT = os.environ.get('APP_PORT', '5000')

# Database
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
