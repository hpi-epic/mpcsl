from hashlib import blake2b

from sqlalchemy.exc import DatabaseError
from werkzeug.exceptions import BadRequest
import networkx as nx

from src.db import db
from src.master.db import data_source_connections
from src.models import Node, EdgeInformation


def load_networkx_graph(result):
    graph = nx.DiGraph(id=str(result.id), name=f'Graph_{result.id}')
    for node in result.job.experiment.dataset.nodes:
        graph.add_node(node.id, label=node.name)
    for edge in result.edges:
        edge_info = EdgeInformation.query.filter_by(edge=edge).one_or_none()
        edge_label = edge_info.annotation.name if edge_info else ''
        graph.add_edge(edge.from_node.id, edge.to_node.id, id=edge.id, label=edge_label, weight=edge.weight)
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
