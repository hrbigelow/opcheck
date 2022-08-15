import tensorflow as tf
import inspect
import traceback
from types import SimpleNamespace
from schema import Schema

# singleton global config object
config = SimpleNamespace(validate = False) 

def validate_schema(do_validate):
    config.validate = do_validate

def register(op_path, init_schema_func, calltime_config_func=None):
    # This section is called 'registration phase'
    mod_path, func_name = op_path.rsplit('.', 1)
    mod = eval(mod_path)
    func = getattr(mod, func_name)
    sig = inspect.signature(func)
    op = Schema(op_path)
    op.init_schema(sig, init_schema_func, calltime_config_func)

    def wrapper(*args, **kwargs):
        # executes during 'framework call phase'
        sig = inspect.signature(func)
        bind_obj = sig.bind(*args, **kwargs)
        bind_obj.apply_defaults()
        bound_args = bind_obj.arguments
        op.p.prepare_call(op, bound_args)

        framework_ex = None

        opcheck_valid = op.p.check_args()
        if opcheck_valid:
            print(f'Opcheck {op.p.op_path} input validation passed.\n')
        else:
            print(f'Opcheck {op.p.op_path} input validation failed.\n\n'
                    f'{op.p.report()}\n')
        try:
            ret_val = func(*args, **kwargs)
        except Exception as ex:
            print(f'Framework op raised exception.\n\n{ex}\n')
            framework_ex = ex
        else:
            op.p.set_outputs(ret_val)
            op.p.check_return()
            """
            if err != '':
                print(f'Opcheck {op.p.op_path} output validation failed.\n\n')
                print(err)
            if msg != '':
                print(f'Opcheck {op.p.op_path} output validation passed.\n\n')
                print(msg)
            """
        finally:
            op.p.report()
            if framework_ex is not None:
                raise framework_ex

        return ret_val
    setattr(mod, func_name, wrapper)

def init():
    import op_schema

