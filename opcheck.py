# tf import is needed for 'eval(mod_path)' 
import tensorflow as tf
import inspect
from schema import Schema

REGISTRY = {}

def register(op_path, init_schema_func, calltime_config_func=None):
    """
    Wrap the framework operation at {op_path} for OpCheck checking.
    {init_schema_func} defines constraints on the input relationships.
    {calltime_config_func} is a function that can be run at call time if
    constraints depend on the inputs.
    """
    # This section is called 'registration phase'
    mod_path, func_name = op_path.rsplit('.', 1)
    mod = eval(mod_path)
    func = getattr(mod, func_name)
    sig = inspect.signature(func)
    op = Schema(op_path)
    op.init_schema(sig, init_schema_func, calltime_config_func)

    def wrapped_op(*args, **kwargs):
        # executes during 'framework call phase'
        sig = inspect.signature(func)
        bind_obj = sig.bind(*args, **kwargs)
        bind_obj.apply_defaults()
        op.p.prepare_call(op, bind_obj.arguments)
        op.p.check_args()
        try:
            ret_val = func(*args, **kwargs)
        except Exception as ex:
            op.p.log_framework_error(ex)
        else:
            op.p.check_return(ret_val)
        finally:
            op.p.report()
            if op.p.framework_error is not None:
                raise op.p.framework_error

        return ret_val
    op.p.wrapped_op = wrapped_op
    setattr(mod, func_name, wrapped_op)
    REGISTRY[op_path] = op

def validate(op_path):
    """
    Run generated test configurations and confirm opcheck flags errors
    appropriately.
    """
    if op_path not in REGISTRY:
        raise RuntimeError(
            f'A tensor op named \'{op_path}\' is not registered with OpCheck. '
            f'Cannot validate.')
    op = REGISTRY[op_path]
    op.p.validate_schema()

def init():
    import ops

