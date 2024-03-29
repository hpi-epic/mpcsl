import json
import os

from src.db import db
from src.master.appfactory import AppFactory
from src.master.helpers.io import InvalidInputData
from src.models import Algorithm, AlgorithmSchema


def set_up_algorithms(db):
    if os.path.isfile('conf/algorithms.json'):
        # The check here is necessary, because without running the migrations,
        # the DB might be empty. To be able to run the migrations though, it is necessary to be able
        # to initialize the app.
        if db.engine.dialect.has_table(db.session, Algorithm.__table__.name):
            with open('conf/algorithms.json') as f:
                algorithms = json.load(f)
                for algorithm in algorithms:
                    data, errors = AlgorithmSchema().load(algorithm)
                    if len(errors) > 0:
                        raise InvalidInputData(payload=errors)
                    alg_in_db = db.session.query(Algorithm).filter(Algorithm.package == data['package'])\
                        .filter(Algorithm.function == data['function']).one_or_none()
                    if not alg_in_db:
                        alg = Algorithm(**data)
                        db.session.add(alg)
                        print(f'==> Function `{data["function"]}` from Package `{data["package"]}` added…')
                    else:
                        alg_in_db.update(data)
                        print(f'==> Function `{data["function"]}` from Package `{data["package"]}` updated…')
                db.session.commit()


if __name__ == "__main__":
    appfactory = AppFactory()

    app = appfactory.migration_up()

    with app.app_context():
        set_up_algorithms(db)
