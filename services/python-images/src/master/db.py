import pyhdb
from sqlalchemy import create_engine

from src.master.config import DATA_SOURCE_CONNECTIONS

pyhdb.protocol.constants.MAX_MESSAGE_SIZE = 4194304  # 2^22 instead of 2^17
pyhdb.protocol.constants.MAX_SEGMENT_SIZE = pyhdb.protocol.constants.MAX_MESSAGE_SIZE - \
    pyhdb.protocol.constants.general.MESSAGE_HEADER_SIZE

data_source_connections = {}
for name, connstring in DATA_SOURCE_CONNECTIONS.items():
    data_source_connections[name] = create_engine(connstring)
