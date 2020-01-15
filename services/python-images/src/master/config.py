import json
import os

API_HOST = os.environ.get('API_HOST')
SCHEDULER_HOST = os.environ.get('SCHEDULER_HOST')
RELEASE_NAME = os.environ.get('RELEASE_NAME', '')
PORT = os.environ.get("PORT")
K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "default")
EXECUTION_IMAGE_NAMESPACE = os.environ.get("EXECUTION_IMAGE_NAMESPACE")

MPCI_ENVIRONMENT = os.environ.get('MPCI_ENVIRONMENT')

# Database
DB_TYPE = os.environ.get('DB_TYPE', 'postgresql')
DB_HOST = os.environ.get('DB_HOST', 'localhost:5432')
DB_DATABASE = os.environ.get('DB_DATABASE', 'postgres')
DB_USER = os.environ.get('POSTGRES_USER', "admin")
DB_PASSWORD = os.environ.get('POSTGRES_PASSWORD', "admin")

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
