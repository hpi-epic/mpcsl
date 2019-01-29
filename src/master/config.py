import os
import json
from configparser import ConfigParser

API_HOST = os.environ.get('API_HOST')

# Database
DB_TYPE = os.environ.get('DB_TYPE', 'postgresql')
DB_HOST = os.environ.get('DB_HOST', 'database')
DB_DATABASE = os.environ.get('DB_DATABASE', 'postgres')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')

# Remote databases for loading
DATA_SOURCE_CONNECTIONS = json.loads(os.environ.get('DATA_SOURCE_CONNECTIONS', '{}'))

SQLALCHEMY_DATABASE_URI = os.environ.get(
    'SQLALCHEMY_DATABASE_URI',
    DB_TYPE + '://' + DB_USER + ':' + DB_PASSWORD + '@' +
    DB_HOST + '/' + DB_DATABASE
)

SQLALCHEMY_TRACK_MODIFICATIONS = os.environ.get(
    'SQLALCHEMY_TRACK_MODIFICATIONS',
    'false'
).lower() == 'true'

DAEMON_CYCLE_TIME = 5


# The following section tries to read the amount of processes from the uWSGI
# configuration file in the root directory of project.
# It can be overridden with an env var to avoid reading the config if necessary.
# This variable is used to determine the process which launches the daemon
# when the server is handled by uWSGI.
# Further information is available in AppFactory::set_up_daemon
UWSGI_NUM_PROCESSES = os.environ.get('UWSGI_NUM_PROCESSES', None)

if UWSGI_NUM_PROCESSES is None:
    uwsgi_conf = ConfigParser()
    uwsgi_conf.read('uwsgi.ini')

    UWSGI_NUM_PROCESSES = int(uwsgi_conf['uwsgi']['processes'])
else:
    UWSGI_NUM_PROCESSES = int(UWSGI_NUM_PROCESSES)
