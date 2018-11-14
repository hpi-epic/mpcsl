from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api

from backend.src.routes import set_up_routes


class AppFactory(object):
    def __init__(self):
        self.app = None
        self.db = None
        self.api = None

    def set_up_db(self):
        self.db = SQLAlchemy(self.app)

    def set_up_app(self):
        self.app = Flask(__name__)
        self.app.config.from_object('backend.src.config')

    def set_up_api(self):
        self.api = Api(self.app)
        set_up_routes(self.api)

    def up(self):
        self.set_up_app()
        self.set_up_api()
        self.set_up_db()
        return self.app
