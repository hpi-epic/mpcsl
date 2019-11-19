import os

from flask import Flask, jsonify
from flask_restful_swagger_2 import Api
from flask_migrate import Migrate
from flask_socketio import SocketIO

from src.db import db
from src.master.helpers.io import InvalidInputData
from .routes import set_up_routes


class AppFactory(object):

    def set_up_db(self):
        self.db = db
        self.db.init_app(self.app)

        self.migrate = Migrate(self.app, db)

    def set_up_app(self):
        self.app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'), static_url_path='/static')
        self.app.config.from_object('src.master.config')

    def set_up_socketio(self):
        if self.app is None:
            raise Exception("Flask app not set")
        self.socketio = SocketIO(self.app)

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

    def set_up_error_handlers(self):
        @self.app.errorhandler(InvalidInputData)
        def handle_invalid_usage(error):
            response = jsonify(error.to_dict())
            response.status_code = error.status_code
            return response

    def up(self):
        self.set_up_app()
        self.set_up_api()
        self.set_up_db()
        self.set_up_error_handlers()
        self.set_up_socketio()
        return [self.app, self.socketio]
