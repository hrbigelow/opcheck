import importlib
import inspect
from schema import SchemaApi
from schema.error import OpCheckInternalError, FrameworkError, Success
from schema.error import NotApplicable

REGISTRY = {}

def register(*op_paths):
    """
    Register each op in {op_paths}, or all available ops if {op_paths} is empty
    """
    if len(op_paths) == 0:
        op_paths = available_ops()

    for op_path in op_paths:
        try:
            _register_op(op_path)
        except BaseException as ex:
            print(f'Got exception: {ex} while registering op '
                    f'\'{op_path}\'.  Skipping.')

def deregister(*op_paths):
    """
    De-register each op in {op_paths}, restoring it back to its original
    un-checked state.
    """
    if len(op_paths) == 0:
        op_paths = available_ops()

    for op_path in op_paths:
        try:
            _deregister_op(op_path)
        except RuntimeError as ex:
            pass

def available_ops():
    """
    List all ops available for registration with OpCheck.  Each op is defined
    in a file in the ops/ directory.
    """
    from pkgutil import walk_packages
    import ops
    modinfos = list(walk_packages(ops.__path__, ops.__name__ + '.'))
    op_paths = [mi.name.split('.',1)[1] for mi in modinfos if not mi.ispkg]
    return op_paths

def _register_op(op_path):
    """
    Wrap the framework operation at {op_path} for OpCheck checking.
    """
    main_mod_name = op_path.split('.')[0]
    func_mod_name, func_name = op_path.rsplit('.',1)

    if main_mod_name == 'tf':
        import tensorflow as tf
        framework_op = eval(op_path)
        framework_mod = eval(func_mod_name) 
    elif main_mod_name == 'torch':
        import torch
        framework_op = eval(op_path)
        framework_mod = eval(func_mod_name) 

    op_qualpath = '.'.join(('ops', op_path))
    schema_mod = importlib.import_module(op_qualpath)
    op = SchemaApi(op_path)
    wrapped_op = op._init_schema(framework_mod, framework_op,
            schema_mod.init_schema)
    setattr(framework_mod, func_name, wrapped_op)
    REGISTRY[op_path] = op

def _deregister_op(op_path):
    op = REGISTRY.pop(op_path, None)
    if op is None:
        raise RuntimeError(
            f'Op path \'{op_path}\' is not registered so cannot be '
            f'de-registered')
    func_name = op_path.rsplit('.',1)[1]
    setattr(op.framework_mod, func_name, op.framework_op)

def _get_from_path(op_path):
    if op_path not in REGISTRY:
        raise RuntimeError(
            f'Could not find an op named \'{op_path}\' in the OpCheck '
            f'registry.  Use opgrind.inventory() to see available ops.')
    op = REGISTRY[op_path]
    return op

def validate(op_path, out_dir, test_ids=None):
    """
    Run generated test configurations and confirm opgrind flags errors
    appropriately, and does not flag errors where none exist.
    """
    op = _get_from_path(op_path)
    op._validate_schema(out_dir, test_ids)

def explain(op_path):
    """
    Produce an explanation of the op
    """
    op = _get_from_path(op_path)
    index_table = op._index_inventory()
    info_table = op._inventory()
    print('\n'.join(index_table))
    print()
    print('\n'.join(info_table))

def registered_ops():
    """
    List all framework ops registered with OpCheck
    """
    return list(REGISTRY.keys())

def _dot_graph(op, nodes, out_file):
    import graphviz
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
    names = { n.name: n.name.replace(':', '_') for n in nodes }
    for node in nodes:
        is_arg = (node in op.arg_gen_nodes.values() or node in
                op.arg_pred_nodes.values())
        color = 'red' if is_arg else 'black'
        dot.node(names[node.name], node.name, color=color)
        for pa in node.parents:
            dot.edge(names[node.name], names[pa.name])
    dot.render(out_file)
    print(f'Wrote {out_file}.pdf')

def print_gen_graph(op_path, out_dir):
    """
    Print a pdf of {op_path} generative graph used for generating test cases 
    """
    op = REGISTRY[op_path]
    nodes = op.gen_graph.values()
    _dot_graph(op, nodes, f'{out_dir}/{op_path}.gen')

def print_pred_graph(op_path, out_dir):
    """
    Print a pdf of {op_path} predicate graph, used for validating inputs to
    the op.
    """
    op = REGISTRY[op_path]
    nodes = op.pred_graph.values()
    _dot_graph(op, nodes, f'{out_dir}/{op_path}.pred')

def print_inventory_graph(op_path, out_dir):
    """
    Print a pdf of {op_path} inventory graph, used for displaying the valid
    inventory (signatures, data format, dtypes) combinations
    """
    op = REGISTRY[op_path]
    nodes = op.inv_graph.values()
    _dot_graph(op, nodes, f'{out_dir}/{op_path}.inv')

