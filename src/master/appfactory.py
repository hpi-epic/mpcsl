import os
import json

from flask import Flask
from flask_restful_swagger_2 import Api

from src.db import db
from .routes import set_up_routes
from src.models import Algorithm, AlgorithmSchema


class AppFactory(object):
    def __init__(self):
        self.app = None
        self.db = None
        self.api = None

    def set_up_db(self):
        self.db = db
        self.db.init_app(self.app)
        with self.app.app_context():
            self.db.create_all()

    def set_up_app(self):
        self.app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'), static_url_path='/static')
        self.app.config.from_object('src.master.config')

    def set_up_api(self):
        self.api = Api(
            self.app,
            api_version='0.0.1',
            api_spec_url='/swagger',
            description='This describes the MPCI backend API. '
                        'The API allows it to define and execute causal '
                        'inference jobs.',
            host=self.app.config['SERVER_NAME'],
            consumes=['application/json'],
            produces=['application/json'],
            title='Causal Inference Pipeline API',
            tags=[
                {
                    'name': 'Experiment',
                    'description': 'All endpoints related to the definition of experiments.'
                },
                {
                    'name': 'Dataset',
                    'description': 'All endpoints related to the definition of datasets.'
                },
                {
                    'name': 'Job',
                    'description': 'Provides information about running and historic jobs.'
                },
                {
                    'name': 'Result',
                    'description': 'Provides execution results and metadata'
                },
                {
                    'name': 'Executor',
                    'description': 'Endpoints used by the worker processes to load data and store results.'
                }
            ]
        )
        set_up_routes(self.api)

    def set_up_algorithms(self):
        if os.path.isfile('conf/algorithms.json'):
            with self.app.app_context():
                with open('conf/algorithms.json') as f:
                    algorithms = json.load(f)
                    for algorithm in algorithms:
                        data, errors = AlgorithmSchema().load(algorithm)
                        if not self.db.session.query(Algorithm).filter(Algorithm.name==data['name']).one_or_none():
                            alg = Algorithm(**data)
                            self.db.session.add(alg)
                self.db.session.commit()

    def up(self):
        self.set_up_app()
        self.set_up_api()
        self.set_up_db()
        self.set_up_algorithms()
        return self.app
