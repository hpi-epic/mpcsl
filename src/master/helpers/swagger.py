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
