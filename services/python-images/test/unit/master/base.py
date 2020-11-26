from unittest import TestCase

from flask import Flask
from flask.ctx import AppContext
from flask.testing import FlaskClient
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

from src.master.appfactory import AppFactory
from src.db import db


class BaseTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factory: AppFactory = AppFactory()
        cls.app: Flask = cls.factory.up()[0]
        cls.api: Api = cls.factory.api
        cls.app_context: AppContext = cls.app.app_context()
        cls.app_context.push()
        cls.test_client: FlaskClient = cls.app.test_client()
        cls.db: SQLAlchemy = db
        cls.original_tables = cls.db.metadata.sorted_tables

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()
        cls.app_context.pop()

    def setUp(self):
        self.db.create_all()

    def tearDown(self):
        self.db.session.remove()
        self.db.reflect()
        self.drop_all()

    def url_for(self, resource, **values):
        adapter = self.app.url_map.bind('localhost:5000')
        return adapter.build(resource.endpoint, values, force_external=True)

    def drop_all(self):
        for tbl in reversed(self.db.metadata.sorted_tables):
            tbl.drop(self.db.engine)
            if tbl not in self.original_tables:
                self.db.metadata.remove(tbl)
