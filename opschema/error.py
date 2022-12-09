class OpSchemaInternalError(BaseException):
    """A bug in OpSchema"""
    def __init__(self, ex):
        self.ex = ex

class SchemaError(BaseException):
    """Represents an error in the Schema definition"""
    def __init__(self, msg):
        self.msg = msg

class SchemaStatus(BaseException):
    """Represent violations of schema constraints"""
    def __init__(self):
        pass

    def message(self, op):
        raise NotImplementedError

class FrameworkError(SchemaStatus):
    def __init__(self, framework_error):
        self.ex = framework_error

    def message(self, op):
        return f'{self.ex}'

