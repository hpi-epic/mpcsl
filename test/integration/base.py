import time
from unittest import TestCase
from multiprocessing import Process

from src.master.appfactory import AppFactory
from src.db import db


class BaseIntegrationTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.factory = AppFactory()
        cls.app = cls.factory.up()
        cls.api = cls.factory.api
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        cls.db = db

        def run_func(app):
            app.run(host="0.0.0.0", port='5000', debug=True)
        cls.app_thread = Process(target=run_func, args=(cls.app, ))

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()

    def setUp(self):
        self.db.create_all()
        self.app_thread.start()
        time.sleep(1)

    def tearDown(self):
        self.app_thread.terminate()
        self.db.session.remove()
        self.db.reflect()
        self.drop_all()

    def drop_all(self):
        con = self.db.session()
        meta = db.metadata
        for table in reversed(meta.sorted_tables):
            con.execute(table.delete())
        con.commit()
