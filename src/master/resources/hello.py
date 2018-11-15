from flask import current_app
from flask_restful import Resource

from db import db
from models import Hello, HelloSchema


class HelloWorld(Resource):
    def get(self):
        # current_app.logger.info('Received request')
        hello = Hello()
        hello.hello = 'world!!!1'
        db.session.add(hello)
        db.session.commit()
        return HelloSchema().dump(hello).data
