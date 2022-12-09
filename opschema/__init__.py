import importlib
import inspect
from . import schema
from . import ops

REGISTRY = {}

def register(*op_paths):
    """
    Register each op in {op_paths}, or all available ops if {op_paths} is empty
    """
    if len(op_paths) == 0:
        op_paths = list_schemas()

    for op_path in op_paths:
        try:
            _register(op_path)
        except BaseException as ex:
            trace = inspect.trace()
            print('Trace:')
            for t in trace:
                print(inspect.getsource(t))
            # print('\n'.join(str(t) for t in trace))
            # trace = inspect.trace()
            # print(inspect.stack())
            print(f'Got exception: {ex} while registering op '
                    f'\'{op_path}\'.  Skipping.')

def deregister(*op_paths):
    """
    De-register each op in {op_paths}, restoring it back to its original
    un-checked state.
    """
    if len(op_paths) == 0:
        op_paths = list_schemas()

    for op_path in op_paths:
        try:
            _deregister(op_path)
        except RuntimeError as ex:
            pass

def list_schemas():
    """
    List all ops available for registration with OpSchema.  Each op is defined
    in a file in the ops/ directory.
    """
    from pkgutil import walk_packages
    from . import ops
    modinfos = list(walk_packages(ops.__path__, ops.__name__ + '.'))
    op_paths = [mi.name.split('.',1)[1] for mi in modinfos if not mi.ispkg]
    return op_paths

def _init_op(op_path):
    op = schema.OpSchema(op_path)
    schema_module = importlib.import_module(f'ops.{op_path}')
    op._init(schema_module.init_schema)
    return op

def _register(op_path):
    """
    Wrap the framework operation at {op_path} for OpSchema checking.
    """
    if op_path in REGISTRY:
        return

    import tensorflow as tf
    func_name = op_path.rsplit('.',1)[1]
    op = _init_op(op_path)
    wrapped_op = op._wrapped()
    setattr(tf, func_name, wrapped_op)
    REGISTRY[op_path] = op

def _unregister(op_path):
    op = REGISTRY.pop(op_path, None)
    if op is None:
        raise RuntimeError(
            f'Op path \'{op_path}\' is not registered so cannot be '
            f'de-registered')
    func_name = op_path.rsplit('.',1)[1]
    setattr(op.framework_mod, func_name, op.framework_op)

def get(op_path):
    """
    Retrieve the op instance from the path given
    """
    if op_path not in REGISTRY:
        raise RuntimeError(
            f'Could not find an op named \'{op_path}\' in the OpSchema '
            f'registry.  Use opschema.inventory() to see available ops, '
            f'and then register chosen ops with opschema.register()')
    op = REGISTRY[op_path]
    return op

def validate(op_path, out_dir, test_ids, skip_ids, dtype_err_quota):
    """
    Run generated test configurations and confirm opschema flags errors
    appropriately, and does not flag errors where none exist.
    """
    op = get(op_path)
    op._set_gen_error_quotas(dtype_err_quota)
    op._validate(out_dir, test_ids)

def list_registered():
    """
    List all framework ops registered with OpSchema
    """
    return list(REGISTRY.keys())

def print_graphs(op_path, out_dir):
    """
    Print pred_graph, gen_graph, comp_graph, and inv_graph of op to
    `out_dir` in pdf format
    """
    op = _init_op(op_path)
    op.print_graphs(out_dir)

def explain(op_path, include_inventory=False):
    """
    Print out a schematic representation of `op_path`
    """
    op = _init_op(op_path)
    print(op.schema_report(include_inventory))

