from marshmallow import fields, validates, ValidationError
from sqlalchemy.ext.mutable import MutableDict

from src.db import db
from src.models.base import BaseModel, BaseSchema


class Algorithm(BaseModel):
    name = db.Column(db.String, unique=True)
    script_filename = db.Column(db.String)
    docker_image = db.Column(db.String)
    description = db.Column(db.String)
    valid_parameters = db.Column(MutableDict.as_mutable(db.JSON))


class AlgorithmSchema(BaseSchema):
    valid_parameters = fields.Dict()

    @validates('valid_parameters')
    def validate_params(self, params):
        # Just for validation
        for key, val in params.items():
            if 'type' not in val or val['type'] not in ['str', 'enum', 'int', 'float', 'bool']:
                raise ValidationError(f'Invalid type for key {key}')
            if not isinstance(val.get('required', False), bool):
                raise ValidationError(f'Field `required` must be bool for key {key}')

            if val['type'] == 'enum':
                if not isinstance(val['values'], list):
                    raise ValidationError(f'Field `values` must be list for key {key}')
                for item in val['values']:
                    if not isinstance(item, str):
                        raise ValidationError(f'Field `values` must be list of strings for key {key}')
            elif val['type'] in ['int', 'float']:
                param_cls = eval(val['type'])
                if not isinstance(val.get('minimum', param_cls()), param_cls):
                    raise ValidationError(f'Field `minimum` must be {param_cls} for key {key}')
                if not isinstance(val.get('maximum', param_cls()), param_cls):
                    raise ValidationError(f'Field `maximum` must be {param_cls} for key {key}')

    class Meta(BaseSchema.Meta):
        model = Algorithm
