import os
import json

API_HOST = os.environ.get('API_HOST')

MPCI_ENVIRONMENT = os.environ.get('MPCI_ENVIRONMENT')

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


RESULT_READ_BUFF_SIZE = int(os.environ.get('RESULT_READ_BUFF_SIZE', 16 * 1024))
RESULT_WRITE_BUFF_SIZE = int(os.environ.get('RESULT_WRITE_BUFF_SIZE', 1024))
LOAD_SEPARATION_SET = os.environ.get('LOAD_SEPARATION_SET', 'false').lower() == 'true'

DOCKER_EXECUTION_NETWORK = os.environ.get('DOCKER_EXECUTION_NETWORK', 'mpci_default')
DOCKER_BASE_URL = os.environ.get('DOCKER_BASE_URL', 'unix://var/run/docker.sock')

DOCKER_MOUNT_LOG_VOLUME = os.environ.get('DOCKER_MOUNT_LOG_VOLUME', 'true').lower() == 'true'
DOCKER_LOG_VOLUME_NAME = os.environ.get('DOCKER_MOUNT_LOG_VOLUME', 'mpci_worker_logs')
DOCKER_LOG_VOLUME_MOUNT_PATH = os.environ.get('DOCKER_LOG_VOLUME_MOUNT_PATH', '/logs')
