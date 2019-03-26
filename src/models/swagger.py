from marshmallow_sqlalchemy import fields as sqlaFields
from marshmallow import fields
from flask_restful_swagger_2 import Schema


TYPE_MAP = {
    fields.String: 'string',
    fields.Integer: 'integer',
    fields.Float: 'number',
    fields.DateTime: 'string',
    sqlaFields.Related: 'integer',
    fields.Field: 'string',
    fields.Raw: 'object',
    fields.Dict: 'object',
    fields.List: 'array',
    fields.Bool: 'boolean',
    sqlaFields.RelatedList: 'array'
}

FORMAT_MAP = {
    fields.String: 'string',
    fields.Integer: 'int64',
    fields.Float: 'float',
    fields.DateTime: 'date-time',
    sqlaFields.Related: 'int64',
    fields.Field: 'string',
    fields.Raw: 'object',
    fields.Dict: 'object',
    fields.List: 'array',
    fields.Bool: 'boolean',
    sqlaFields.RelatedList: 'array'
}

PYTHON_TYPE_MAP = {
    bool: 'boolean',
    float: 'number',
    str: 'string',
    int: 'integer'
}

PYTHON_FORMAT_MAP = {
    bool: 'boolean',
    float: 'float',
    str: 'string',
    int: 'int64'
}

SWAGGER_SCHEMATA = {}


class SwaggerMixin(object):
    SwaggerSchema = {}

    @classmethod
    def include_field(cls, fieldname, field):
        if not hasattr(cls, 'Meta'):
            return True

        dump_only = cls.Meta.dump_only if hasattr(cls.Meta, 'dump_only') else []
        exclude = cls.Meta.exclude if hasattr(cls.Meta, 'exclude') else []

        return not field.dump_only and fieldname not in dump_only + exclude

    @classmethod
    def get_swagger(cls, for_load=False):
        op = 'Load' if for_load else 'Dump'
        if cls.__name__ + op in cls.SwaggerSchema:
            return cls.SwaggerSchema[cls.__name__ + op]

        properties = {}
        for fieldname, field in cls._declared_fields.items():
            if type(field) != sqlaFields.Related and \
                    (not for_load or cls.include_field(fieldname, field)) and \
                                (field is not None):
                if type(field) == fields.Nested:
                    definition = field.schema.get_swagger()
                    if field.many:
                        definition = definition.array()
                elif type(field) == fields.List:
                    definition = {
                        'type': 'array',
                        'items': {
                            'type': TYPE_MAP[type(field.container)],
                            'format': FORMAT_MAP[type(field.container)]
                        }
                    }
                elif type(field) == sqlaFields.RelatedList:
                    definition = {
                        'type': 'array',
                        'items': {
                            'type': 'integer',
                            'format': 'int64'
                        }
                    }
                elif type(field) == fields.Constant:
                    definition = {
                        'type': PYTHON_TYPE_MAP[type(field.constant)],
                        'format': PYTHON_FORMAT_MAP[type(field.constant)]
                    }
                else:
                    definition = {
                        'type': TYPE_MAP[type(field)],
                        'format': FORMAT_MAP[type(field)]
                    }
                properties[fieldname] = definition

        class CustomSchema(Schema):
            type = 'object'
            properties = {}
        CustomSchema.__name__ = cls.__name__ + (op if op != 'Dump' else '')
        CustomSchema.properties = properties

        cls.SwaggerSchema[cls.__name__ + op] = CustomSchema

        return CustomSchema
