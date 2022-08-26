# tf import is needed for 'eval(mod_path)' 
import tensorflow as tf
import inspect
from schema import SchemaApi
from schema.error import FrameworkError, Success, NotApplicable

REGISTRY = {}

def register(op_path, init_schema_func):
    """
    Wrap the framework operation at {op_path} for OpCheck checking.
    {init_schema_func} defines constraints on the input relationships.
    """
    # This section is called 'registration phase'
    mod_path, func_name = op_path.rsplit('.', 1)
    mod = eval(mod_path)
    func = getattr(mod, func_name)
    sig = inspect.signature(func)
    op = SchemaApi(op_path)
    op._init_schema(sig, init_schema_func)

    def wrapped_op(*args, **kwargs):
        # executes during 'framework call phase'
        sig = inspect.signature(func)
        bind_obj = sig.bind(*args, **kwargs)
        bind_obj.apply_defaults()
        op._prepare_call(bind_obj.arguments)
        op._check_args()
        try:
            ret_val = func(**bind_obj.arguments)
        except Exception as ex:
            op._log_framework_status(ex)
            op.return_status = NotApplicable()
        else:
            op.framework_status = Success()
            op._check_return(ret_val)
        finally:
            print('in finally:  framework status: ',
                    op.framework_status.message(op))
            # assert(op.p.framework_status is not None)
            op._report()
            if isinstance(op.framework_status, FrameworkError):
                raise op.framework_status.ex
            return ret_val
    op.wrapped_op = wrapped_op
    setattr(mod, func_name, wrapped_op)
    REGISTRY[op_path] = op

def validate(op_path):
    """
    Run generated test configurations and confirm opcheck flags errors
    appropriately, and does not flag errors where none exist.
    """
    if op_path not in REGISTRY:
        raise RuntimeError(
            f'A tensor op named \'{op_path}\' is not registered with OpCheck. '
            f'Cannot validate.')
    op = REGISTRY[op_path]
    op._validate_schema()

def init():
    import ops

