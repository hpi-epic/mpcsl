import os
from sys import argv

from flask import Flask, jsonify
from flask_restful_swagger_2 import Api
from flask_migrate import Migrate
from werkzeug.serving import is_running_from_reloader

from src.db import db
from src.master.config import UWSGI_NUM_PROCESSES
from src.master.helpers.daemon import JobDaemon
from src.master.helpers.io import InvalidInputData
from .routes import set_up_routes


class AppFactory(object):
    def __init__(self):
        self.app = None
        self.db = None
        self.api = None
        self.migrate = None
        self.daemon = None

    def set_up_db(self):
        self.db = db
        self.db.init_app(self.app)

        self.migrate = Migrate(self.app, db)

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

    def set_up_daemon(self, force=False):
        """
        This function starts the daemon in one of three cases:
        - force is set to True
        - the last command of the launch command was server.py, which means
          that the server was launched by launching server.py directly
        - the last command was uWSGI and the current pid % UWSGI_NUM_PROCESSES is zero
          the last check is necessary, to ensure that the daemon is only
          launched in a single uWSGI worker. uWSGI workers have pids that are sequential
          up in a docker container, for example 4,5,6,7 so only 1 of them % num workers equals
          zero.
        :param force: Bool, set to true to force daemon launch.
        :return:
        """
        if force or (not is_running_from_reloader()
                     and (argv[-1] == 'server.py'
                     or (argv[-1] == 'uwsgi' and os.getpid() % UWSGI_NUM_PROCESSES == 0))):
            self.app.logger.info("Starting daemon.")
            self.daemon = JobDaemon(self.app, name='Job-Daemon', daemon=True)
            self.daemon.start()

    def set_up_error_handlers(self):
        @self.app.errorhandler(InvalidInputData)
        def handle_invalid_usage(error):
            response = jsonify(error.to_dict())
            response.status_code = error.status_code
            return response

    def up(self, no_daemon=False):
        self.set_up_app()
        self.set_up_api()
        self.set_up_db()
        self.set_up_error_handlers()
        if not no_daemon:
            self.set_up_daemon()
        return self.app
