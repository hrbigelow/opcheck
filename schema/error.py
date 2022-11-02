from collections import namedtuple

class OpGrindInternalError(BaseException):
    """A bug in OpGrind"""
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

class Success(SchemaStatus):
    def __init__(self):
        pass

    def message(self, op):
        return 'Success'

class FrameworkError(SchemaStatus):
    def __init__(self, framework_error):
        self.ex = framework_error

    def message(self, op):
        return f'{self.ex}'

class ArgValueError(SchemaStatus):
    def __init__(self, arg_name, arg_val):
        self.arg_name = arg_name
        self.arg_val = arg_val

    def message(self, op):
        msg = (f'Argument \'{self.arg_name}\' received invalid value '
                f'{self.arg_val}')
        return msg

class ComponentConstraintError(SchemaStatus):
    """
    Returned by an index dims constraint function.  The contents are then
    incorporated into an IndexConstraintError
    """
    def __init__(self, text, error_mask):
        self.text = text
        self.mask = error_mask

class ReturnShapeError(SchemaStatus):
    """The output at {out_idx} does not match the shape implied by its
    signature"""
    def __init__(self, out_idx):
        self.idx = out_idx

    def message(self, op):
        msg = f'The output at {self.idx} does not match the shape predicted '
        msg += f'by its signature'
        return msg

class OutputNumberMismatch(SchemaStatus):
    """The number of outputs returned differed from expected"""
    def __init__(self, exp_num_outputs, act_num_outputs):
        self.expected_num_outputs = exp_num_outputs
        self.actual_num_outputs = act_num_outputs

    def message(self, op):
        msg = f'Expected {self.expected_num_outputs} but got '
        msg += f'{self.actual_num_outputs}'
        return msg

