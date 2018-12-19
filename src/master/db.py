from sqlalchemy import create_engine

from src.master.config import DATA_SOURCE_CONNECTIONS

data_source_connections = {}
for name, connstring in DATA_SOURCE_CONNECTIONS:
    data_source_connections[name] = create_engine(connstring)
