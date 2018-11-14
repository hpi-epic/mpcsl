from flask import Flask
from flask_restful import Api

from src.db import db
from .routes import set_up_routes


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
        self.app = Flask(__name__)
        self.app.config.from_object('src.master.config')

    def set_up_api(self):
        self.api = Api(self.app)
        set_up_routes(self.api)

    def up(self):
        self.set_up_app()
        self.set_up_api()
        self.set_up_db()
        return self.app