from src.master.appfactory import AppFactory
from src.db import db
from src.master.resources.experiments import Experiment
from src.master.resources.datasets import Dataset


def add_experiment(db, dataset_id):
    new_experiment = Experiment(alpha=0.9, cores=1,
                                independence_test="", dataset_id=dataset_id)
    db.session.add(new_experiment)
    db.session.commit()


def add_dataset(db):
    new_dataset = Dataset(name="Porsche", load_query="")
    db.session.add(new_dataset)
    db.session.commit()
    return new_dataset.id


if __name__ == "__main__":
    appfactory = AppFactory()

    app = appfactory.up()

    with app.app_context():
        dataset_id = add_dataset(db)
        add_experiment(db, dataset_id)
