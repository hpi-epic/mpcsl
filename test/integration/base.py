import os
import signal
import time
from unittest import TestCase
from multiprocessing import Process
from urllib.error import URLError
from urllib.request import urlopen

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
        cls.original_tables = cls.db.metadata.sorted_tables

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()

    def setUp(self):
        def run_func(app):
            app.run(host="0.0.0.0", port='5000', debug=True, use_reloader=False, threaded=True)
        self.app_thread = Process(target=run_func, args=(self.app, ))

        self.db.create_all()

        self.app_thread.start()
        timeout = 5
        while timeout > 0:
            time.sleep(1)
            try:
                urlopen('localhost:5000')
                timeout = 0
            except URLError:
                timeout -= 1

    def tearDown(self):
        self.stop_app_thread()
        self.db.session.remove()
        self.db.reflect()
        self.drop_all()

    def drop_all(self):
        for tbl in reversed(self.db.metadata.sorted_tables):
            tbl.drop(self.db.engine)
            if tbl not in self.original_tables:
                self.db.metadata.remove(tbl)

    def stop_app_thread(self):
        if self.app_thread:
            if self._stop_cleanly():
                return
            if self.app_thread.is_alive():
                self.app_thread.terminate()

    def _stop_cleanly(self, timeout=5):
        try:
            os.kill(self.app_thread.pid, signal.SIGINT)
            self.app_thread.join(timeout)
            return True
        except Exception as ex:
            print('Failed to join the live server process: {}'.format(ex))
            return False

    def url_for(self, resource, **values):
        adapter = self.app.url_map.bind('localhost:5000')
        return adapter.build(resource.endpoint, values, force_external=True)
