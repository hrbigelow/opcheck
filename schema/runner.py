import tensorflow as tf
import numpy as np
import io, os
import inspect
import traceback
from .redirect import stderr_redirector
from .error import OpGrindInternalError
from . import fgraph, oparg, generators as ge
from tensorflow.python.framework import ops

class TestResult(object):
    """
    One schema test result.  Holds all sufficient information about the test.
    Provides convenient reporting and classification functions
    """
    def __init__(self, op, _id, arg_map, index_ranks, gen_edit):
        self.op = op
        self.id = _id
        self.arg_map = arg_map
        self.index_ranks = index_ranks

        # list of errors (expected to be zero or one) incurred during the
        # generation of the test example
        self.gen_edit = gen_edit

        # list of suggested fixes.  each item in the list is an error state
        # the list could be empty if no suggested fixes could be found
        self.suggestions = []
        self.framework_error = None
        self.framework_msg = None

    def __repr__(self):
        msg = (f'id: {self.id}\n'
                f'gen_edit: {self.gen_edit}\n'
                f'suggestions: {self.suggestions}\n'
                f'framework_error: {self.framework_error}\n'
                f'framework_msg: {self.framework_msg}\n'
                )
        return msg

    def add_result(self):
        self.suggestions = self.op.input_errors
        if self.op.framework_error is None:
            self.framework_error = self.op.framework_error
        else:
            # holding a reference to the actual framework exception here
            # results in GPU tensor memory not being freed 
            self.framework_error = type(self.op.framework_error.ex)
            self.framework_msg = str(self.op.framework_error.ex)

        if len(self.suggestions) == 0:
            cat = 'FAIL'
        else:
            top_sug = self.suggestions[0]
            z = zip(top_sug.infos, self.gen_edit.infos)
            if len(top_sug.infos) != len(self.gen_edit.infos):
                cat = 'FAIL'
            elif any(t.obj != g.obj for t, g in z):
                cat = 'FAIL'
            else:
                fr_neg = (self.framework_error is None)
                if len(self.gen_edit.infos) == 0:
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

        nsug = len(self.suggestions)
        if nsug == 0:
            opgrind_err = 'No Hits'
        else:
            opgrind_err = repr(self.suggestions[0])

        if self.framework_error is None:
            framework_error_str = 'Success'
        else:
            framework_error_str = self.framework_error.__name__

        stats = [ 
                str(self.id), 
                self.category, 
                repr(self.gen_edit),
                opgrind_err,
                framework_error_str,
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
        op_msg = repr(self.suggestions)
        fr_msg = self.framework_msg

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
    Uses the gen_graph to produce a list of (<target_error>, <arg_dict>)
    pairs for the op.  <target_error> is the correct type of SchemaStatus.
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
    configs = op._generate_tests() 
    print(f'Generated {len(configs)} tests')

    # Create the test instances
    tests = []
    for test_id, (stat, args, ranks) in enumerate(configs, 1):
        t = TestResult(op, test_id, args, ranks, stat) 
        tests.append(t)

    test_id = 1
    run_number = 0
    mem = { 'current': 0 }
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
            t.run()
            # pre = mem
            # mem = tf.config.experimental.get_memory_info('GPU:0')
            # delta = mem['current'] - pre['current']
            # print(f'Test {t.id:-4d}: {delta}', flush=True)
            # continue

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
