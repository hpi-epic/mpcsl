from hashlib import blake2b

from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest
import networkx as nx

from src.db import db
from src.master.db import data_source_connections
from src.models import Node


def load_networkx_graph(result):
    node_list = [node.id for node in result.job.experiment.dataset.nodes]
    edge_list = [(edge.from_node_id, edge.to_node_id) for edge in result.edges]
    graph = nx.DiGraph(node_list, edge_list)
    return graph


def check_dataset_hash(dataset):
    session = get_db_session(dataset)

    try:
        result = session.execute(dataset.load_query).fetchone()
        num_of_obs = session.execute(f"SELECT COUNT(*) FROM ({dataset.load_query}) _subquery_").fetchone()[0]
    except DatabaseError:
        raise BadRequest(f'Could not execute query "{dataset.load_query}" on database "{dataset.data_source}"')

    hash = blake2b()
    concatenated_result = str(result) + str(num_of_obs)
    hash.update(concatenated_result.encode())

    return str(hash.hexdigest()) == dataset.content_hash


def add_dataset_nodes(dataset):
    session = get_db_session(dataset)

    try:
        result = session.execute(dataset.load_query).fetchone()
    except DatabaseError:
        raise BadRequest(f'Could not execute query "{dataset.load_query}" on database "{dataset.data_source}"')

    for key in result.keys():
        node = Node(name=key, dataset=dataset)
        db.session.add(node)
        db.session.commit()


def get_db_session(dataset):
    if dataset.data_source != "postgres":
        session = data_source_connections.get(dataset.data_source, None)
        if session is None:
            raise BadRequest(f'Could not reach database "{dataset.data_source}"')
    else:
        session = db.session
    return session
