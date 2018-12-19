from sqlalchemy import create_engine

from src.master.config import DB_CONNECTIONS

extension_dbs = {}
for name, connstring in DB_CONNECTIONS:
    extension_dbs[name] = create_engine(connstring)
