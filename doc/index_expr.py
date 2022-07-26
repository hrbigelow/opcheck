# Demonstration of index expression unrolling
"""
An index is an identifier used inside array brackets.  In this context, it
is assumed that the dimension (i.e. the highest value the index can obtain,
plus 1), is a property associated with the index primarily, not the array.

An index expression is an arithmetic expression built using indices.

In this tutorial, the notion of 'unrolling' an index, list of indices, or index
expression, is demonstrated.

Then, using a target np.array, the index expression values are mapped to cell
values.

I introduce some new terminology here.  The first is the term 'index expression
basis'.  This is the dimensions of the array obtained after unrolling the index
expression.  These dimensions are determined as the cartesian product of the
dimensions of the set of index identifiers in the expression.

The unrolled 
"""
import numpy as np

# evaluate index_expr across the cartesian product of index values
# return the evaluations as an array
def unroll(index_expr, dims):
    ufunc = np.frompyfunc(index_expr, nin=len(dims), nout=1)
    return np.fromfunction(ufunc, dims, dtype=int)

# create a new array mapping each ind in unrolled_inds to its value in ary
# returning 0 for out of bounds indices
def map_ary(unrolled_inds, ary):
    def pyfunc(idx):
        return ary[idx] if idx in np.ndindex(ary.shape) else 0
    ufunc = np.vectorize(pyfunc)
    return ufunc(unrolled_inds)

source = np.array([1,4,7,5,2])
grid = np.array([
        [1,1,2,1,1],
        [1,3,3,4,4],
        [2,3,9,6,5],
        [1,6,4,4,3],
        [1,1,3,1,1]
        ])


idim = 8
jdim = 8
basis = (idim, jdim)

print(f'Unrolling the index expression \'j-i+2\'\n')
inds = unroll(lambda i,j: (j-i+2,), basis)
print(inds)
print('\n')

print(f'The source array\n')
print(source)
print('\n')

print('Evaluating array expression \'source[j-i+2]\'\n')
vals = map_ary(inds, source)
print(vals)
print('\n')

print(f'Unrolling the index expression \'(i-1,j-2)\'\n')
inds = unroll(lambda i,j: (i-1,j-2), basis)
print(inds)
print('\n')

print(f'The grid array\n')
print(grid)
print('\n')

print(f'Evaluating array expression \'grid[i-1,j-2]\'\n')
vals = map_ary(inds, grid)
print(vals)
print('\n')





