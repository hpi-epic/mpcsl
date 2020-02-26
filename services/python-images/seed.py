import pandas as pd
import networkx as nx

from src.db import db
from src.master.appfactory import AppFactory
from src.master.helpers.database import add_dataset_nodes
from src.models import Algorithm, Experiment, Dataset, Edge


def add_experiment(db, dataset_id):
    alg = db.session.query(Algorithm).filter(Algorithm.package == 'pcalg').first()
    new_experiment = Experiment(
        algorithm_id=alg.id,
        dataset_id=dataset_id,
        name="Earthquake experiment",
        description="with pcalg and binCI",
        parameters={
            'alpha': 0.01,
            'cores': 1,
            'independence_test': 'binCI',
            'verbose': 1
        }
    )

    db.session.add(new_experiment)
    db.session.commit()


def add_dataset(db):
    df = pd.read_csv('test/fixtures/earthquake_10k.csv', index_col=0) \
        .astype('category').apply(lambda c: c.cat.codes, axis=0)
    df.to_sql('test_data', con=db.engine, index=False, if_exists="replace")
    db.session.commit()

    new_dataset = Dataset(
        name="Earthquake dataset",
        description="10k observations x 5 nodes",
        load_query="SELECT * FROM test_data",
        data_source="postgres"
    )
    db.session.add(new_dataset)

    add_dataset_nodes(new_dataset)

    return new_dataset.id


def add_ground_truth_edges(db, dataset_id):
    graph = nx.read_gml('test/fixtures/earthquake_groundtruth.gml')
    ds = Dataset.query.get(dataset_id)
    for edge in graph.edges:
            from_node_label = edge[0]
            to_node_label = edge[1]
            from_node_index = None
            to_node_index = None

            for node in ds.nodes:
                if from_node_label == node.name:
                    from_node_index = node.id
                if to_node_label == node.name:
                    to_node_index = node.id
            edge = Edge(result_id=None, from_node_id=from_node_index,
                        to_node_id=to_node_index, weight=None, is_ground_truth=True)
            db.session.add(edge)
            db.session.commit()


if __name__ == "__main__":
    appfactory = AppFactory()

    app = appfactory.migration_up()

    with app.app_context():
        print("==> Seeding Test Data")
        dataset_id = add_dataset(db)
        add_ground_truth_edges(db, dataset_id)
        add_experiment(db, dataset_id)
