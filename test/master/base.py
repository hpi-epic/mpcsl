from unittest import TestCase

from src.master.appfactory import AppFactory
from src.db import db


class BaseTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factory = AppFactory()
        cls.app = cls.factory.up()
        cls.api = cls.factory.api
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        cls.test_client = cls.app.test_client()
        cls.db = db

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

    def drop_all(self):
        con = self.db.session()
        meta = db.metadata
        for table in reversed(meta.sorted_tables):
            con.execute(table.delete())
        con.commit()
