from db import db
from models.base import BaseModel, BaseSchema


class Hello(BaseModel):
    hello = db.Column(db.String, default="World")


class HelloSchema(BaseSchema):
    class Meta:
        model = Hello
