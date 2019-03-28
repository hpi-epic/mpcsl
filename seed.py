import pandas as pd

from src.master.appfactory import AppFactory
from src.db import db
from src.models import Algorithm, Experiment, Dataset
from src.master.helpers.database import add_dataset_nodes


def add_experiment(db, dataset_id):
    alg = db.session.query(Algorithm).filter(Algorithm.name == 'pcalg').first()
    new_experiment = Experiment(
        algorithm_id=alg.id,
        dataset_id=dataset_id,
        name="Example experiment",
        description="This is an example description",
        parameters={
            'alpha': 0.01,
            'cores': 1,
            'independence_test': 'binCI'
        }
    )

    db.session.add(new_experiment)
    db.session.commit()


def add_dataset(db):
    df = pd.read_csv('test/fixtures/earthquake_10k.csv', index_col=0) \
        .astype('category').apply(lambda c: c.cat.codes, axis=0)
    df.to_sql('test_data', con=db.engine, index=False)
    db.session.commit()

    new_dataset = Dataset(name="Example dataset", load_query="SELECT * FROM test_data", data_source="postgres")
    db.session.add(new_dataset)

    add_dataset_nodes(new_dataset)

    return new_dataset.id


if __name__ == "__main__":
    appfactory = AppFactory()

    app = appfactory.up(no_daemon=True)

    with app.app_context():
        dataset_id = add_dataset(db)
        add_experiment(db, dataset_id)
