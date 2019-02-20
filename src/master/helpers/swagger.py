from marshmallow import fields, Schema

from src.models.swagger import SwaggerMixin


def get_default_response(response_model):
    return {
        '200': {
            'description': 'Success',
            'schema': response_model,
        },
        '404': {
            'description': 'Object not found'
        },
        '400': {
            'description': 'Invalid input'
        },
        '500': {
            'description': 'Internal server error'
        }
    }


def oneOf(schemas):
    field_names = [f'Option #{i+1} - {schema.__name__}' for i, schema in enumerate(schemas)]

    class OneOfSchema(Schema, SwaggerMixin):
        for field_name, schema in zip(field_names, schemas):
            vars()[field_name] = fields.Nested(schema)
        del field_name
        del schema

    return OneOfSchema
