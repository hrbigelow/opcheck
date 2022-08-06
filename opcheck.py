import tensorflow as tf
import inspect
import traceback
from types import SimpleNamespace
from schema import OpSchema

# singleton global config object
config = SimpleNamespace(validate = False) 

def validate_schema(do_validate):
    config.validate = do_validate

def register(op_path):
    mod_path, func_name = op_path.rsplit('.', 1)
    mod = eval(mod_path)
    func = getattr(mod, func_name)
    op = OpSchema(op_path)

    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bind_obj = sig.bind(*args, **kwargs)
        bind_obj.apply_defaults()
        bound_args = bind_obj.arguments
        op.init(bound_args)
        opcheck_valid = op.evaluate()
        # even if opcheck_valid is False, still run the framework op.
        # then, compare the error messages
        try:
            ret_val = func(*args, **kwargs)
        except Exception as e:
            tf_error = traceback.format_exc()
            print(f'Error from TensorFlow:\n\n{e}\n\n')
            print(f'Error from opcheck:\n\n{op.report()}\n\n')
            raise RuntimeError('Opcheck')
        if config.validate:
            err, msg = op.validate(ret_val)
            if err != '':
                print(err)
            if msg != '':
                print(msg)

        return ret_val
    
    setattr(mod, func_name, wrapper)
    return op

def init():
    import op_schema


class Broadcastable(object):
    # allows integer and integer list values to
    # broadcast in binary arithmetic operations
    def __init__(self, int_or_intlist):
        self.val = int_or_intlist
    
    def _op(self, b, op):
        pass

    def __add__(self, o):
        pass

    def __sub__(self, o):
        pass

    def __mul__(self, o):
        pass

    def __div__(self, o):
        pass
