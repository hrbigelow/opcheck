import io, os
import inspect
from .redirect import stderr_redirector
from .error import Success, OpGrindInternalError
from . import fgraph, oparg, generators as ge

class TestResult(object):
    """
    One schema test result.  Holds all sufficient information about the test.
    Provides convenient reporting and classification functions
    """
    def __init__(self, op, _id, arg_map, index_ranks, status_list):
        self.op = op
        self.id = _id
        self.arg_map = arg_map
        self.index_ranks = index_ranks
        err = list(s for s in status_list if s != Success)
        if len(err) > 1:
            stat = None
        else:
            stat = err[0] if len(err) > 0 else Success
        self.expect_class = stat

    def add_result(self):
        self.opgrind_status = self.op.input_status
        if isinstance(self.op.framework_status, Success):
            self.framework_status = self.op.framework_status
        else:
            self.framework_status = self.op.framework_status.ex

        ex_stat_cls = self.expect_class
        op_stat = self.opgrind_status
        fr_stat = self.framework_status
        if type(op_stat) != ex_stat_cls:
            cat = 'FAIL'
        else:
            op_neg = isinstance(op_stat, Success)
            fr_neg = isinstance(fr_stat, Success)
            if op_neg:
                cat = 'TN' if fr_neg else 'FN'
            else:
                cat = 'FP' if fr_neg else 'TP'
        self.category = cat

    def make_args(self):
        # create arguments.  generative node values are either OpArg instances
        # which generate the value, or are taken as-is.
        arg_dict = {}
        for arg_name, node_val in self.arg_map.items():
            if isinstance(node_val, oparg.OpArg):
                arg_val = node_val.value()
            else:
                arg_val = node_val
            arg_dict[arg_name] = arg_val
        return arg_dict

    def stat_keys(self):
        # return an ordered set of keys
        keys = ['ID', 'CATEGORY', 'EXPECT_STATUS', 'OPGRIND_STATUS',
                'FRAMEWORK_STATUS', 'RANKS']
        keys.extend(self.op.arg_order)
        return keys

    def stats(self):
        # summary statistics for the test
        stats = []
        keys = self.stat_keys()

        stats = [ 
                str(self.id), 
                self.category, 
                self.expect_class.__name__,
                self.opgrind_status.__class__.__name__,
                self.framework_status.__class__.__name__,
                ','.join(str(r) for r in self.index_ranks.values())
                ]

        for arg_name in self.op.arg_order:
            node_val = self.arg_map.get(arg_name, None)
            if node_val is None:
                continue
            if isinstance(node_val, oparg.OpArg):
                rep = node_val.summary()
            else:
                rep = str(node_val)
            stats.append(rep)
        return stats

    def run(self):
        # run the op and store the results
        string_err = io.BytesIO()
        arg_dict = self.make_args()

        try:
            with stderr_redirector(string_err):
                self.op.wrapped_op(**arg_dict)
        except OpGrindInternalError as e:
            print(string_err.getvalue().decode('UTF-8'))
            raise e
        except BaseException as e:
            # this should always be from TensorFlow
            trace = inspect.trace()
            for frame in reversed(trace):
                mod = inspect.getmodule(frame[0])
                if mod is not None:
                    break
            modname = mod.__name__
            # print(modname, flush=True)
            if modname.split('.')[0] == 'tensorflow':
                # print('exception inside tensorflow:')
                # traceback.print_stack()
                pass
            else:
                assert False, 'exception outside tf should not be possible'
                print('exception outside of tensorflow')
                traceback.print_stack()
                raise e
        self.add_result()

    def report(self):
        """
        Generate a human-readable report
        """
        cat = self.category
        op_stat = self.opgrind_status
        op_msg = op_stat.message(self.op)
        fr_msg = self.op.framework_status.message(self.op)
        if cat == 'TP':
            return f'Framework\n{fr_msg}\nOpGrind\n{op_msg}\n'
        elif cat == 'TN':
            return ''
        elif cat == 'FP':
            return f'OpGrind\n{op_msg}\n'
        elif cat == 'FN':
            return f'Framework\n{fr_msg}\n'
        elif cat == 'FAIL':
            return f'OpGrind\n{op_msg}\n'
        else:
            raise RuntimeError(f'unknown test category {cat}')


def validate(op, out_dir, test_ids=None, skip_ids=None):
    """
    Uses the gen_graph to produce a list of (<target_status>, <arg_dict>)
    pairs for the op.  <target_status> is the correct type of SchemaStatus.
    Runs the wrapped op on the <arg_dict> and collects the actual
    SchemaStatus and any possible exception from the framework.

    If {test_ids} is provided, it is an iterable of integers which specify
    a subset of tests to run.  This is useful for speeding up debugging.
    """
    if not os.path.exists(out_dir):
        raise RuntimeError(
            f'{type(op).__qualname__}: Could not open output path '
            f'\'{out_dir}\' for report generation')

    if skip_ids is None:
        skip_ids = set()

    # list of successful arg_dict settings
    key_order = [ 'TP', 'TN', 'FP', 'FN', 'FAIL' ]
    suffixes = ['stats'] + [k for k in key_order]
    stats = { 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'FAIL': [] }

    path = lambda sfx: os.path.join(out_dir, f'{op.op_path}.{sfx}.txt')
    files = { sfx: path(sfx.lower()) for sfx in suffixes }

    print('Generating tests')
    # list of node.name, value
    test_id = 1
    tests = []
    config_list = list(fgraph.gen_graph_iterate(*op.gen_graph.values()))
    stat_name = fgraph.node_name(ge.StatusAggregator)
    ranks_name = fgraph.node_name(ge.GetRanks)

    print(f'Generated {len(config_list)} candidates')
    for node_vals in config_list:
        arg_vals = {}
        for arg_name in op.arg_order:
            node = op.arg_gen_nodes.get(arg_name, None)
            if node is None:
                continue
            val = node_vals[node.name]
            arg_vals[arg_name] = val

        status_list = node_vals[stat_name]
        index_ranks = node_vals[ranks_name]
        t = TestResult(op, test_id, arg_vals, index_ranks, status_list)
        test_id += 1
        if t.expect_class is None:
            test_id -= 1
            continue
        tests.append(t)

    print(f'Created {len(tests)} tests')
    print()

    test_id = 1
    with open(files['TP'], 'w') as tp, \
            open(files['TN'], 'w') as tn, \
            open(files['FP'], 'w') as fp, \
            open(files['FN'], 'w') as fn, \
            open(files['FAIL'], 'w') as fail, \
            open(files['stats'], 'w') as stats_fh:
        fh = { 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'FAIL':
                fail, 'stats': stats_fh }
        is_first_line = True
        for t in tests:
            skip = False
            if test_ids is not None:
                if len(test_ids) == 0:
                    break
                if t.id not in test_ids:
                    skip = True
                else: 
                    test_ids.remove(t.id)
            if t.id in skip_ids:
                skip = True

            if skip:
                mat = ', '.join(f'{c}: {len(stats[c])}' for c in key_order)
                print('\r', end='')
                print(f'Skipping test: {t.id:-4d}  Stats: {mat}', end='')
                continue

    
            print('\r', end='')
            print(f'Running test: {t.id:-4d}  ', end='')
            arg_dict = t.make_args()
            t.op._prepare_call(**arg_dict)
            t.op._check_args()
            continue
            t.run()
            cat = t.category
            row = t.stats()
            stats[cat].append(t)
            mat = ', '.join(f'{c}: {len(stats[c])}' for c in key_order)
            print(f'Stats: {mat}', end='')

            if is_first_line:
                hdr = t.stat_keys()
                hdr_line = '\t'.join(h for h in hdr)
                print(hdr_line, file=fh['stats'], flush=True)
                is_first_line = False

            line = '\t'.join(r for r in row)
            print(line, file=fh['stats'], flush=True)
            print(line, file=fh[cat], flush=True)
            print(t.report(), file=fh[cat], flush=True)

    print()
    print('Summary')
    for cat in key_order:
        res = stats[cat]
        print(f'{cat}: {len(res)}')
