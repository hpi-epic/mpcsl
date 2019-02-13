from hashlib import blake2b

from src.db import db
from src.master.db import data_source_connections


def check_dataset_hash(dataset):
    if dataset.remote_db is not None:
        session = data_source_connections.get(dataset.remote_db, None)
        if session is None:
            return None
    else:
        session = db.session

    result = session.execute(dataset.load_query)

    result = result.fetchall()

    hash = blake2b()
    hash.update(result)

    return hash.hexdigest() == dataset.content_hash

