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
            if not op._passed():
                op._report()
            # print('in finally:  framework status: ',
                    # op.framework_status.message(op))
            # assert(op.p.framework_status is not None)
            if isinstance(op.framework_status, FrameworkError):
                raise op.framework_status.ex
            return ret_val
    op.wrapped_op = wrapped_op
    setattr(mod, func_name, wrapped_op)
    REGISTRY[op_path] = op

def _get_from_path(op_path):
    if op_path not in REGISTRY:
        raise RuntimeError(
            f'Could not find an op named \'{op_path}\' in the OpCheck '
            f'registry.  Use opcheck.inventory() to see available ops.')
    op = REGISTRY[op_path]
    return op

def validate(op_path):
    """
    Run generated test configurations and confirm opcheck flags errors
    appropriately, and does not flag errors where none exist.
    """
    op = _get_from_path(op_path)
    op._validate_schema()

def explain(op_path):
    """
    Produce an explanation of the op
    """
    op = _get_from_path(op_path)
    index_table = op._index_inventory()
    info_table = op._sig_inventory()
    print('\n'.join(index_table))
    print()
    print('\n'.join(info_table))

def inventory():
    print('\n'.join(REGISTRY.keys()))

def init():
    import ops

