import numpy as np

from src.master.appfactory import AppFactory
from src.db import db
from src.master.resources.experiments import Experiment
from src.master.resources.datasets import Dataset


def add_experiment(db, dataset_id):
    new_experiment = Experiment(
        dataset_id=dataset_id,
        name="Example experiment",
        description="This is an example description",
        parameters={
            'alpha': 0.9,
            'cores': 1,
            'independence_test': 'gaussCI'
        }
    )

    db.session.add(new_experiment)
    db.session.commit()


def add_dataset(db):
    new_dataset = Dataset(name="Example dataset", load_query="SELECT * FROM test_data")
    db.session.add(new_dataset)

    db.session.execute("""
        CREATE TABLE IF NOT EXISTS test_data (
            a float,
            b float,
            c float
        );
    """)

    mean = [0, 5, 10]
    cov = [[1, 0, 0], [0, 10, 0], [0, 0, 20]]
    source = np.random.multivariate_normal(mean, cov, size=50)
    for l in source:
        db.session.execute("INSERT INTO test_data VALUES ({0})".format(",".join([str(e) for e in l])))
    db.session.commit()

    return new_dataset.id


if __name__ == "__main__":
    appfactory = AppFactory()

    app = appfactory.up()

    with app.app_context():
        dataset_id = add_dataset(db)
        add_experiment(db, dataset_id)
