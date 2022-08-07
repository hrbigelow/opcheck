import tensorflow as tf
import numpy as np
import itertools
import re
import util
from parse import BCParser
from collections import defaultdict
from ast_nodes import EinTup, IntExpr, Dims, ArithmeticBinOp, StaticExpr

class Runtime(object):
    def __init__(self, reps, min_dim, max_dim):
        # tf.config.list_physical_devices('GPU')
        self.parser = BCParser() 
        # map of eintup names to EinTup instances
        self.tups = {}

        # anonymous eintups used for rank constraints 
        self.anon_tups = []

        # defines the signature (see notes.txt) of arrays
        self.array_sig = {}

        # stores current values of the arrays.  the shape
        # either matches or is broadcastable to the signature
        self.arrays = {}

        # The program statement top-level AST nodes
        self.statements = None 

        # Ast nodes representing rank and dim comstraints
        self.constraints = None

        self.reps = reps
        self.min_dim = IntExpr(self, min_dim)
        self.max_dim = IntExpr(self, max_dim) 
        self.parser.set_runtime(self)

    def __repr__(self):
        name_pad = max(len(name) for name in self.arrays.keys())
        tups = 'Tups: \n' + '\n'.join(repr(tup) for tup in self.tups.values())

        sigs = 'Array Signatures: \n' 
        sigs += '\n'.join(name + ': ' + repr(sig) for name, sig in
                self.array_sig.items())

        shapes = 'Array Shapes: \n'
        for name, sig in self.array_sig.items():
            shapes += f'{name:{name_pad+3}s}:'
            shapes += ', '.join(repr(elem.dims()) for elem in sig)
            shapes += '\n'

        statements = 'Statements: \n'
        statements += '\n'.join(repr(st) for st in self.statements)

        tfcall = 'TF Call: \n'
        tfcall += repr(self.tf_call)

        out_args = 'Output Args: \n'
        out_args += repr(self.out_args)

        return (f'{tups}\n\n{sigs}\n\n{shapes}\n\n{statements}\n\n'
                f'{tfcall}\n\n{out_args}\n')

    def parse_et_file(self, et_file):
        with open(et_file, 'r') as fh:
            content = fh.read()

        sections = iter(re.split('\n\n+', content.strip()))
        statements = next(sections)
        tf_call = next(sections)
        tf_output = next(sections)
        constraints = next(sections, '')

        statements = statements.strip().split('\n')
        tf_call = tf_call.replace('\n', ' ').strip()
        tf_output = tf_output.replace('\n', ' ').strip()
        constraints = constraints.strip().split('\n')

        self.parser.set_statement_mode()
        self.statements = [ self.parser.parse(st) for st in statements ]

        self.parser.set_tfcall_mode()
        self.tf_call = self.parser.parse(tf_call)

        # ordered list of TensorArg nodes in the order matching expected tf
        # output
        self.parser.set_output_mode()
        self.out_args = self.parser.parse(tf_output)
        
        self.parser.set_constraint_mode()
        for con in constraints:
            self.parser.parse(con)

        # post-init all AST nodes
        for tup in self.tups.values():
            tup.lift_rank_range()

        for tup in self.anon_tups:
            tup.lift_rank_range()

        # add implicit rank constraint of 1 to any unconstrained EinTups
        for tup in self.tups.values():
            tup.maybe_set_rank_range(range(1,2))

        all_nodes = self.statements + [self.tf_call]

    def clear_shapes(self):
        for tup in self.tups.values():
            tup.clear()

    def init_all_shapes(self, tups, shapes):
        for tup, shape in zip(tups, shapes):
            tup.initialize(shape)
        self.arrays.clear()

    # generate shapes according to ordered tups
    def gen_shapes(self, tups, reps=30):
        range_tups = [ t for t in tups if t.rank_range is not None ]
        range_list = [ t.rank_range for t in range_tups ]
        combos = itertools.product(*range_list)

        for ranks in combos:
            for i in range(reps):
                self.clear_shapes()
                for t, r in zip(range_tups, ranks):
                    t.set_rank(r)
                for t in tups:
                    t.calc_rank()
                for t in tups:
                    t.gen_dims()
                shapes = [ t.dims() for t in tups ]
                yield shapes

    def validate_all(self):
        vis_tups = list(self.tups.values())
        all_tups = vis_tups + self.anon_tups
        n_vis = len(vis_tups)

        # generates all combinations of conformant shapes and ranks for tups
        all_shapes = list(self.gen_shapes(all_tups, self.reps))
        # compute padding
        width = [ len(t.name) for t in vis_tups ]
        for shapes in all_shapes:
            for i in range(n_vis):
                name = vis_tups[i].name
                width[i] = max(width[i], len(str(shapes[i])))

        print(''.join(f'{vis_tups[i].name:<{width[i]+3}s}' 
            for i in range(n_vis)), '  Valid')

        for shapes in all_shapes:
            self.init_all_shapes(all_tups, shapes)
            valid = self.validate()
            line = ''.join(f'{str(shapes[i]):<{width[i]+3}s}' for i in
                    range(n_vis))
            print(f'{line}   {valid}')

    # validate the current rank + dims setting
    def validate(self):
        for st in self.statements:
            st.evaluate()
        tf_outputs = self.tf_call.value()
        z = zip(self.out_args, tf_outputs)
        valid = [ util.equal_tens(et.value(), tf_out, 1e-6) for et, tf_out in z ]
        return valid

    def maybe_add_tup(self, name):
        if name not in self.tups:
            self.tups[name] = EinTup(name)
        return self.tups[name]

    def add_anon_tup(self, tup):
        self.anon_tups.append(tup)

    def tup(self, eintup):
        if eintup not in self.tups:
            raise RuntimeError(
                    f'Runtime::tup() got unknown eintup name {eintup}')
        return self.tups[eintup]

    def dims(self, eintup):
        return self.tup(eintup).dims()

    def rank(self, eintup):
        return len(self.dims(eintup))

    def nelem(self, eintup):
        return self.tup(eintup).nelem()

if __name__ == '__main__':
    rt = Runtime()
    rt.set_ranks({'batch': 2, 'slice': 1})

