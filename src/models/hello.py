from src.db import db
from src.models.base import BaseModel, BaseSchema


class Hello(BaseModel):
    hello = db.Column(db.String, default="World")


class HelloSchema(BaseSchema):
    class Meta:
        model = Hello
