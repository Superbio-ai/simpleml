from server.resources.controller import CrossValidate


def init_routes(api, session_controller):
    api.add_resource(CrossValidate, '/api/cross_validate',
                     resource_class_kwargs={'session_controller': session_controller})
    