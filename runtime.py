import tensorflow as tf
import numpy as np
import itertools
import re
from parse import BCParser
from ast_nodes import IntExpr, Dims, ArithmeticBinOp

def equal_tens(a, b, eps):
    if not a.dtype.is_floating:
        eps = 0
    return (
            a.shape == b.shape and
            tf.reduce_all(tf.less_equal(tf.abs(a - b), eps)).numpy()
            )

def maybe_broadcast(a, length):
    if isinstance(a, (list, tuple)):
        if len(a) != length:
            raise RuntimeError(
                f'Cannot broadcast {a} to length {length}')
        else:
            return a
    else:
        return [a] * length


class Shape(object):
    # simple data class
    def __init__(self, min_expr, max_expr):
        self.dims = None
        self.rank = None
        self.min_exprs = [min_expr]
        self.max_exprs = [max_expr]

    def __repr__(self):
        return (f'Shape: rank {self.rank}, dims {self.dims}, ' 
                f'mins: {self.min_exprs}, maxs: {self.max_exprs}')

    # two Shapes are considered neighbors if there exists a constraint
    # with Dims on either side.  The set of all such Dims-Dims constraints
    # cannot produce a cycle among the induced graph of all Shapes.
    def dims_neighbors(self): 
        exprs = self.min_exprs + self.max_exprs
        return { dn for ex in exprs for dn in ex.get_nodes_of_type(Dims) }

    # recursively generate dimensions for this and any neighbors.
    # neighbor connections are defined by 
    def gen_dims(self):
        if self.rank is None:
            raise RuntimeError(
                f'Cannot call Shape::gen_dims() before rank is set')

        if self.has_dims():
            return

        def is_ready(expr):
            dnodes = expr.get_nodes_of_type(Dims)
            return all(tup.has_dims() for d in dnodes for tup in d.base_tups)

        # check that all Dims-containing constraints match the rank
        for nb in self.dims_neighbors():
            if any(tup.rank() != self.rank for tup in nb.base_tups):
                raise RuntimeError(
                    f'Dims constraint {nb} contains one or more EinTups with '
                    f'rank differing from this shape\'s rank {self.rank}')

        min_expr = self.min_exprs[0]
        for ex in self.min_exprs[1:]:
            if is_ready(ex):
                min_expr = ArithmeticBinOp(min_expr, ex, 'max')
        max_expr = self.max_exprs[0]
        for ex in self.max_exprs[1:]:
            if is_ready(ex):
                max_expr = ArithmeticBinOp(max_expr, ex, 'min')
        try:
            min_vals = min_expr.value()
            max_vals = max_expr.value()
        except RuntimeError as rt:
            raise RuntimeError(
                f'Shape::gen_dims has inconsistent ranked constraints. '
                f'{rt.value}')
        min_vals = maybe_broadcast(min_vals, self.rank)
        max_vals = maybe_broadcast(max_vals, self.rank)

        z = zip(min_vals, max_vals)
        dims = [ np.random.randint(lo, hi) for lo, hi in z ]
        self.dims = dims

        assert len(self.dims) == self.rank, (
                f'gen_dims {self.dims} does not match rank {self.rank}')

        for nbor in self.dims_neighbors():
            for tup in nbor.base_tups:
                tup.gen_dims()

    def set_rank(self, rank):
        self.dims = None
        self.rank = rank

    def get_rank(self):
        if self.rank is None:
            raise RuntimeError(
                f'Cannot call Shape::get_rank() before rank is set')
        return self.rank

    def _add_limit_expr(self, expr, is_max):
        for dn in expr.get_nodes_of_type(Dims):
            if len(dn.base_tups) != 1:
                raise RuntimeError(
                    f'Only single-EinTup Dims expressions allowed '
                    f'in constraints. Got {dn}')
        if is_max:
            self.max_exprs.append(expr)
        else:
            self.min_exprs.append(expr)

    def add_max_expr(self, max_expr):
        return self._add_limit_expr(max_expr, True)

    def add_min_expr(self, min_expr):
        return self._add_limit_expr(min_expr, False)

    def has_dims(self):
        return self.dims is not None

    def set_elem(self, ind, dim):
        if self.dims is None:
            raise RuntimeError('Cannot call set_elem() on uninitialized Shape')
        if ind >= len(self.dims):
            raise RuntimeError(
                f'set_elem() index {ind} out of bounds for length '
                f'{len(self.dims)} dims')
        self.dims[ind] = dim

    def get_dims(self):
        if self.dims is None:
            raise RuntimeError('Cannot call get_dims() on uninitialized Shape')
        return self.dims


class EinTup(object):
    def __init__(self, name, min_expr, max_expr, shadow_of=None):
        self.name = name
        self.shadow_of = shadow_of
        if shadow_of is None:
            self.shape = Shape(min_expr, max_expr) 
        else:
            self.shape = shadow_of.shape
        self._value = None

    def __repr__(self):
        try:
            dimstring = ','.join([str(d) for d in self.dims()])
        except RuntimeError:
            dimstring = '?'
        try:
            rankstring = self.rank()
        except RuntimeError:
            rankstring = '?'
        shadow = ''
        if not self.primary():
            shadow = f'(shadowing {self.shadow_of.name})'
        return f'EinTup \'{self.name}\' |{rankstring}| [{dimstring}]'

    def __len__(self):
        return len(self.dims())

    def __iter__(self):
        self.index = np.ndindex(*self.dims())
        return self

    def __next__(self):
        # intentionally silent.  simply used to advance the position
        self._value = next(self.index)
        return self.value()
    
    def primary(self):
        return self.shadow_of is None

    def same_shape_as(self, other):
        return self.shape is other.shape 

    def set_rank(self, rank):
        self.shape.set_rank(rank)

    def maybe_add_max_expr(self, max_expr):
        if self.primary():
            self.shape.add_max_expr(max_expr)

    def maybe_add_min_expr(self, min_expr):
        if self.primary():
            self.shape.add_min_expr(min_expr)

    def gen_dims(self):
        if self.shadow_of is not None:
            raise RuntimeError(f'cannot call set_dims on shadowing EinTup')
        self.shape.gen_dims()

    def has_dims(self):
        return self.shape.has_dims()

    def dims(self):
        return self.shape.get_dims()

    def rank(self):
        return self.shape.get_rank()

    def nelem(self):
        return np.prod(self.dims(), dtype=np.int32)

    def value(self):
        if self._value is None:
            raise RuntimeError(f'{self} called value() before iteration')
        return self._value


class Runtime(object):
    def __init__(self, min_dim=5, max_dim=100):
        self.parser = BCParser() 
        # map of eintup names to EinTup instances
        self.tups = {}

        # defines the signature (see notes.txt) of arrays
        self.array_sig = {}

        # stores current values of the arrays.  the shape
        # either matches or is broadcastable to the signature
        self.arrays = {}

        # The program statement top-level AST nodes
        self.statements = None 

        # Ast nodes representing rank and dim comstraints
        self.constraints = None

        self.min_dim = IntExpr(self, min_dim)
        self.max_dim = IntExpr(self, max_dim) 
        self.parser.set_runtime(self)

    def __repr__(self):
        tups = 'Tups: \n' + '\n'.join(repr(tup) for tup in self.tups.values())

        sigs = 'Array Signatures: \n' 
        sigs += '\n'.join(name + ': ' + repr(sig) for name, sig in
                self.array_sig.items())

        shapes = 'Array Shapes: \n'
        shapes += '\n'.join(name + ': ' + repr(ary.shape) 
                for name, ary in self.arrays.items())

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
        cons = [ con for s in constraints for con in self.parser.parse(s) ]
        def dims_con(con):
            return isinstance(con.arg1, Dims) or isinstance(con.arg2, Dims) 

        self.rank_constraints = [ con for con in cons if not dims_con(con) ]
        self.dims_constraints = [ con for con in cons if dims_con(con) ]

        # post-init all AST nodes
        all_nodes = (
                self.statements + self.rank_constraints +
                self.dims_constraints  + [self.tf_call])

        for node in all_nodes: 
            node.post_parse_init()

        self.register_dims_limits()

    def register_dims_limits(self):
        # add Dims constraints to appropriate EinTups
        def plus1(expr):
            return ArithmeticBinOp(expr, IntExpr(self, 1), '+')

        all_ops = ['<','<=','==','>=','>']
        for con in self.dims_constraints:
            flipped_op = dict(zip(all_ops, reversed(all_ops)))[con.op_string]
            g1 = (con.op_string, con.arg1, con.arg2)
            g2 = (flipped_op, con.arg2, con.arg1)
            for op, lhs, rhs in g1, g2:
                min_expr = max_expr = None
                if isinstance(lhs, Dims):
                    if op == '<':
                        max_expr = rhs
                    elif op == '<=':
                        max_expr = plus1(rhs)
                    elif op == '==':
                        min_expr = rhs 
                        max_expr = plus1(rhs) 
                    elif op == '>=':
                        min_expr = rhs 
                    elif op == '>':
                        min_expr = plus1(rhs) 
                if min_expr is not None:
                    for tup in lhs.base_tups:
                        tup.maybe_add_min_expr(min_expr)
                if max_expr is not None:
                    for tup in lhs.base_tups:
                        tup.maybe_add_max_expr(max_expr)
        
    # run the full program and produce the set of output tensors in the
    # preconfigured order
    def run(self):
        if not all(con.value() for con in self.constraints):
            return None
        for st in self.statements:
            st.evaluate()
        outs = { (arg.name, arg.value()) for arg in self.outputs }
        return outs

    def gen_dims(self):
        for tup in self.tups.values():
            if not tup.primary() or tup.has_dims():
                continue
            tup.gen_dims()

    # cycle through all combinations of ranks < 10 satisfying constraints
    def cycle(self, k):
        cons = self.rank_constraints
        if k == -1:
            yield 
            return
        pre_tups = set(tup.name for con in cons[:k] for tup in con.get_tups())
        cur_tups = set(tup.name for tup in cons[k].get_tups())
        extra = list(cur_tups.difference(pre_tups))
        for _ in self.cycle(k-1):
            for cur_ranks in np.ndindex((10,) * len(extra)):
                update = dict(zip(extra, cur_ranks))
                self.set_ranks(update)
                if cons[k].value():
                    yield
        
    def validate_all(self):
        k = len(self.rank_constraints) - 1
        keys = list(self.tups.keys())
        key_header = '\t'.join(keys)
        print(f'{key_header}\tValidated')
        for _ in self.cycle(k):
            # config = [ tup.rank() for tup in self.tups.values() ]
            self.gen_dims()
            config = '\t'.join([ repr(self.tups[k].dims()) for k in keys ])
            valid = self.validate()
            print(f'{config}\t{valid}')

    # validate the current rank + dims setting
    def validate(self):
        for st in self.statements:
            st.evaluate()
        tf_outputs = self.tf_call.value()
        z = zip(self.out_args, tf_outputs)
        valid = [ equal_tens(et.value(), tf_out, 1e-6) for et, tf_out in z ]
        return valid

    def set_ranks(self, rank_map):
        for tup, rank in rank_map.items():
            if tup not in self.tups:
                raise RuntimeError('Cannot set dims for unknown EinTup {tup}')
            if not self.tup(tup).primary():
                raise RuntimeError(
                    f'Cannot set rank for non-primary EinTup {tup}')
            self.tup(tup).set_rank(rank)

    def set_dims(self, dims_map):
        for name, dims in dims_map.items():
            self.tups[name].set_dims(dims)

    def set_one_dim(self, tup, ind, val):
        self.tup(tup).set_dim(ind, val)

    def maybe_add_tup(self, name, shadow_of=None):
        if name in self.tups:
            pass
        elif shadow_of is None:
            self.tups[name] = EinTup(name, self.min_dim, self.max_dim, None)
        elif shadow_of.name in self.tups:
            self.tups[name] = EinTup(name, self.min_dim, self.max_dim, shadow_of)
        else:
            raise RuntimeError(
                f'Runtime::maybe_add_tup - shadow_of \'{shadow_of}\' '
                f'provided but does not exist')
        return self.tups[name]

    def get_primary_tups(self):
        return [ tup for tup in self.tups.values() if tup.primary() ]

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

