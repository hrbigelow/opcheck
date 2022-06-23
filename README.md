# Tuple Einstein Notation Mini-language 

A language for expressing Tensorflow operations in a rank-agnostic way

## Synopsis

```bash
# Run an op defined in Tuple Einstein notation on all allowed sub-rank
# combinations of inputs.  Check for validity against the Tensorflow op output.

python run_op.py ops/gather_nd.json
python run_op.py ops/scatter_nd.json
python run_op.py ops/matmul.json
...
```

## Motivation

Many TensorFlow operations work across a combination of ranks for batch,
spatial, indexing and other dimensions.  For instance, `convolution` works for
input with 1, 2, or 3 spatial dimensions.  Batched matrix multiplication works
for a number of batch dimensions from 0 to at least 9.  `gatherNd` works with
variable numbers of dimensions for batching, slice layout and slice
dimensions.

Though the internal logic of the operations is often very clear, there isn't a
straightforward way to define it in a rank-agnostic way.  This language
attempts to provide a definition.

There are several benefits to having such a language.  Once it is comprehended,
the definition serves as a precise documentation of its behavior and allowed
inputs.  Second, it can serve as a contract for unit tests.  Third, having such
a clear language might encourage new op proposals to be communicated.

## Definition of Tuple Einstein notation

Tuple Einstein notation is an extension of Einstein summation notation in which
a group of Einstein indices in a fixed order (a tuple) is given a name, like
`batch` or `slice`, for example.  The expression is then "compiled" by first
choosing a rank (and dimensions) (i.e. a *shape*) for each tuple.  Then, each
statement is evaluated on the full combination of eintup values.  There are
other constructs introduced which will be explained later.  For example, here
is the definition for `gatherNd`:

`ops/gather_nd.json`
```json
{
  "program": [
    "params[batch,elem] = RANDOM()",
    "indices[batch,slice,coord] = RANDINT(0, DIMS(elem)[coord])",
    "result[batch,slice] = params[batch,indices[batch,slice,:]]"
  ],
  "constraints": [
    "RANK(batch) + RANK(slice) < 8",
    "RANK(batch) + RANK(elem) < 8",
    "RANK(elem) > 0",
    "RANK(coord) == 1",
    "DIMS(coord)[0] == RANK(elem)"
  ],
  "tfcall": {
    "func": "tf.gather_nd",
    "args": {
      "params": "params",
      "indices": "indices",
      "batch_dims": "RANK(batch)"
    },
    "return-value": "result"
  }
}
```

The definition is provided by the `program` field in three lines.  Every line
is a single assignment statement in which the left-hand side is a scalar access
of an array.  The right-hand-side is either a function call producing a scalar,
or an array element access producing a scalar.  Array indices are a
comma-separated list of the eintups.  

In the above example, the eintups are `batch`, `elem`, `slice`, and `coord`.
The function `RANK(batch)` gives the number of Einstein indices in `batch`.
`DIMS(batch)` is an array whose elements are the number of values, i.e. the
*shape*.  `batch` takes on all value combinations produced by
`np.ndindex(DIMS(batch))`.  For example, suppose `RANK(batch) = 3` and
`DIMS(batch) = [2,4,3]`.  Then, `batch` would take on the combination of values
in `[0, 2) x [0,4), x [0,3)`

Non-scalar array access may be used to provide the values for an eintup, but
not at the top level.  In the right-hand side of the third statement,
`params[batch,indices[batch,slice,:]]`, the `:` takes the place of `coord`, and
`indices[batch,slice,:]` provides the value for the `elem` eintup in the
`params` array.  This is possible because of the provided constraints
`RANK(coord) == 1` and `DIMS(coord)[0] == RANK(elem)`, so the slice has the
appropriate number of values.

The two special functions `RANDOM()` and `RANDINT()` symbolize
`np.random.uniform` and `np.random.randint`, respectively.  They are used both
to initialize arrays, and implicitly define their type as float or integer.

