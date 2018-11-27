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

        def run_func(app):
            app.run(host="0.0.0.0", port='5000', debug=True, use_reloader=False, threaded=True)
        cls.app_thread = Process(target=run_func, args=(cls.app, ))

    @classmethod
    def tearDownClass(cls):
        cls.db.engine.dispose()

    def setUp(self):
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
        con = self.db.session()
        meta = db.metadata
        for table in reversed(meta.sorted_tables):
            con.execute(table.delete())
        con.commit()

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
