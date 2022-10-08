# tf import is needed for 'eval(mod_path)' 
import importlib
import inspect
from schema import SchemaApi
from schema.error import OpCheckInternalError, FrameworkError, Success
from schema.error import NotApplicable

REGISTRY = {}

def init(*op_names):
    if len(op_names) == 0:
        from pkgutil import walk_packages
        import ops
        modinfos = list(walk_packages(ops.__path__, ops.__name__ + '.'))
        op_names = [mi.name.split('.',1)[1] for mi in modinfos if not mi.ispkg]
    for op_name in op_names:
        try:
            register(op_name)
        except BaseException as ex:
            print(f'Got exception: {ex} while registering op '
                    f'\'{op_name}\'.  Skipping.')

def register(op_name):
    """
    Wrap the framework operation at {op_path} for OpCheck checking.
    {init_schema_func} defines constraints on the input relationships.
    """
    # This section is called 'registration phase'
    op_path = '.'.join(('ops', op_name))
    mod = importlib.import_module(op_path)
    sig = inspect.signature(mod.init_schema)
    op = SchemaApi(op_path)
    op._init_schema(sig, mod.init_schema)

    def wrapped_op(*args, **kwargs):
        # executes during 'framework call phase'
        try:
            op._prepare_call(*args, **kwargs)
            op._check_args()
        except BaseException as ex:
            raise OpCheckInternalError(ex)
        try:
            ret_val = func(**op.arguments)
            # ret_val = None
        except BaseException as ex:
            op.framework_status = FrameworkError(ex)
            op.return_status = NotApplicable()
        else:
            op.framework_status = Success()
            op._check_return(ret_val)
        finally:
            if not op._passed():
                op._report()
            if isinstance(op.framework_status, FrameworkError):
                raise op.framework_status.ex
            return ret_val
    op.wrapped_op = wrapped_op
    setattr(mod, op_name, wrapped_op)
    REGISTRY[op_name] = op

def _get_from_path(op_path):
    if op_path not in REGISTRY:
        raise RuntimeError(
            f'Could not find an op named \'{op_path}\' in the OpCheck '
            f'registry.  Use opcheck.inventory() to see available ops.')
    op = REGISTRY[op_path]
    return op

def validate(op_path, out_dir, test_ids=None):
    """
    Run generated test configurations and confirm opcheck flags errors
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

def list_ops():
    """
    List all framework ops registered with OpCheck
    """
    print('\n'.join(REGISTRY.keys()))

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

