from hashlib import blake2b
from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest

from src.db import db
from src.master.db import data_source_connections
from src.models import Node


def check_dataset_hash(dataset):
    if dataset.remote_db is not None:
        session = data_source_connections.get(dataset.remote_db, None)
        if session is None:
            raise BadRequest(f'Could not reach database "{dataset.remote_db}"')
    else:
        session = db.session

    try:
        result = session.execute(dataset.load_query).fetchone()
    except DatabaseError:
        raise BadRequest(f'Could not execute query {dataset.load_query} on database "{dataset.remote_db}"')

    hash = blake2b()
    hash.update(str(result).encode())

    return str(hash.hexdigest()) == dataset.content_hash


def add_dataset_nodes(dataset):
    if dataset.remote_db is not None:
        session = data_source_connections.get(dataset.remote_db, None)
        if session is None:
            raise BadRequest(f'Could not reach database "{dataset.remote_db}"')
    else:
        session = db.session

    try:
        result = session.execute(dataset.load_query).fetchone()
    except DatabaseError:
        raise BadRequest(f'Could not execute query {dataset.load_query} on database "{dataset.remote_db}"')

    for key in result.keys():
        node = Node(name=key, dataset=dataset)
        db.session.add(node)
        db.session.commit()
