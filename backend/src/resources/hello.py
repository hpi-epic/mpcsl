from flask_restful import Resource

from backend.src.db import db
from backend.src.models.hello import Hello, HelloSchema


class HelloWorld(Resource):
    def get(self):
        hello = Hello()
        hello.hello = 'world!!!1'
        db.session.add(hello)
        db.session.commit()
        return HelloSchema().dump(hello).data
