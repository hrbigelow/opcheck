
# Einsum Tuple - a mini-language for defining tensor operations

## Motivation

Most Tensorflow and PyTorch operations are defined in terms of *dimension
subspaces*.  For example, most operations involve the 'batch dimensions'.
Convolutions have the notion of 'spatial dimensions', 'input channels' and
'output channels'.  The depthwise separable convolution, and the
`tf.nn.space_to_depth` operation have the notion of 'depth'.

Many of these operations work across a range of choices for the *rank* of these
dimension subspaces.  Convolutions for instance work across the spatial
dimensions ranks of 1, 2, or 3.

To take a more complex example,
[tf.gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd) 
is described without explicit names for these groups of dimensions, other than
the 'batch dimensions'.  The documentation describes the notion of [gathering
scalars](https://www.tensorflow.org/api_docs/python/tf/gather_nd#gathering_scalars)
and [gathering
slices](https://www.tensorflow.org/api_docs/python/tf/gather_nd#gathering_scalars)
as two separate subcases.  But, this is hard to visualize without explicit
names for these dimension subspaces.

Here, I propose a mini-language, **Einsum Tuple**, to define such tensor
operations concisely.  In most cases, after defining the inputs, the actual
logic of the operation can be expressed in a single assignment statement.  The
language's central notion is the *eintup* - a tuple-form of an Einstein
summation index which names a dimension subspace.  The length of the *eintup*
represents the rank of the subspace, and is unspecified in the expression,
making the expressions generic with respect to ranks.

## Examples

### Convolution for any spatial dimensions 

```python
# Valid Convolution (for any number of spatial dimensions)
# Excerpt from ops/conv_valid_v1.et and ops/conv_valid_v2.et
input[batch,ipos,ichan] = RANDOM(0, 10, FLOAT)
filters[fpos,ichan,ochan] = RANDOM(0, 1, FLOAT)
output[batch,opos,ochan] = filters[ipos-DIMS(stride)*opos,ichan,ochan] * input[batch,ipos,ichan]

# Alternate definition, showing linearity with respect to filters
output[batch,opos,ochan] = filters[fpos,ichan,ochan] * input[batch,fpos+DIMS(stride)*opos,ichan]

# Validation call
tf.nn.convolution(input=input, filters=filters, strides=DIMS(stride), padding=L('VALID'))
```

To validate the Einsum Tuple definition for convolution against
`tf.nn.convolution`, run the `.et` file.  The runtime system instantiates the
*Einsum Tuple* program with all valid combinations of ranks of Eintups, and
runs each instance.

```bash
# Validate the above expression for 1D, 2D and 3D
$ python eintup.py ops/conv_valid_v1.et
batch   ipos           ichan   fpos        ochan   opos        stride         Valid
[3]     [24]           [2]     [4]         [1]     [11]        [2]            [True]
[1]     [24, 15]       [3]     [2, 5]      [2]     [12, 11]    [2, 1]         [True]
[1]     [18, 18, 20]   [1]     [2, 3, 4]   [3]     [6, 8, 6]   [3, 2, 3]      [True]
```

### Gather

```python
# The Gather operation
params[batch,elem,other] = RANDOM(0, 10, FLOAT)
indices[batch,slice,coord] = RANDOM(0, DIMS(elem)[coord], INT)
result[batch,slice,other] = params[batch,indices[batch,slice,:],other]

# Validation
tf.gather_nd(params, indices, batch_dims=RANK(batch))
```

Here there are many possible valid rank combinations for the Eintups `batch1`,
`elem`, `slice`, `coord`, and `other`.

```python
$ python eintup.py ops/gather_nd.et
batch       elem                   other          slice          coord      Valid
[]          [13]                   []             []             [1]        [True]
[]          [5]                    []             [14]           [1]        [True]
[]          [5]                    []             [12, 2]        [1]        [True]
[]          [6]                    []             [2, 9, 10]     [1]        [True]
[]          [14]                   [8]            []             [1]        [True]
[]          [2]                    [16]           [11]           [1]        [True]
[]          [6]                    [11]           [18, 7]        [1]        [True]
[]          [2]                    [16]           [11, 14, 18]   [1]        [True]
[]          [3]                    [6, 16]        []             [1]        [True]
[]          [8]                    [6, 5]         [3]            [1]        [True]
[]          [18]                   [18, 2]        [6, 8]         [1]        [True]
[]          [5]                    [6, 10]        [19, 18, 20]   [1]        [True]
[]          [11]                   [18, 16, 12]   []             [1]        [True]
[]          [1]                    [6, 12, 11]    [14]           [1]        [True]
[]          [17]                   [15, 14, 14]   [5, 13]        [1]        [True]
[]          [8]                    [5, 8, 6]      [8, 9, 1]      [1]        [True]
[]          [10, 14]               []             []             [2]        [True]
[]          [13, 16]               []             [7]            [2]        [True]
...
```

### Meshgrid

Einsum Tuple allows a straightforward expression of broadcasting behavior.  Any
Eintups which appear on the left hand side but not the right are automatically,
broadcasted, in the same manner as Einstein Summation.  Using this notation,
the last four lines of the Einsum Tuple program provide a concise description
of meshgrid's coordinated broadcasting behavior.

```python
# Excerpt from ops/meshgrid.et
in1[a] = RANDOM(0, 10, INT)
in2[b] = RANDOM(0, 10, INT)
in3[c] = RANDOM(0, 10, INT)
in4[d] = RANDOM(0, 20, INT)
out1[a,b,c,d] = in1[a]
out2[a,b,c,d] = in2[b]
out3[a,b,c,d] = in3[c]
out4[a,b,c,d] = in4[d]

# Validate against tf meshgrid call.  The call returns an array of 
tf.meshgrid(in1, in2, in3, in4, indexing=L('ij'))
```

```bash
$ python eintup.py ops/meshgrid.et
a      b      c      d         Valid
[96]   [92]   [35]   [76]      [True, True, True, True]
```

### Space to Depth

Einsum Tuple has the notion of an *index expression* - an arithmetic expression
of Eintups and integers used to index a tensor.  In this example, the
[tf.nn.space_to_depth](https://www.tensorflow.org/api_docs/python/tf/nn/space_to_depth)
operation can be expressed in a single line.  Note however that the Einsum
Tuple definition given below is actually a more generic operation.
`tf.nn.space_to_depth` is a specialization of this:

1. `bsz` must be rank 2 and square shaped.
2. `ichan` and `batch` must be rank 1.

In the expression, the `output` has four dimension subspaces.  The first and
last are indexed by `batch` and `ichan` respectively.  The middle two are
*index expressions* and comprise the main logic of `tf.nn.space_to_depth`.

Index expressions are component-wise, so the ranks of each binary operation
must match.  The `FLAT()` function is described in more detail in the companion
article.  See
[ops/flatten.et](https://github.com/hrbigelow/einsum-tuple/blob/master/ops/flatten.et)
for a simple example of its usage.

```python
# From ops/space_to_depth.et
input[batch,ipos,ichan] = RANDOM(0, 100, INT)
output[batch, ipos//DIMS(bsz), FLAT(ipos % DIMS(bsz), ichan)] = input[batch,ipos,ichan]

# Validate against Tensorflow op
tf.nn.space_to_depth(input, block_size=L(2), data_format=L('NHWC'))
```

To validate, run `eintup.py` on the `.et` definition file.

```bash
$ python eintup.py ops/space_to_depth.et
batch   ipos     ichan   bsz         Valid
[1]     [4, 4]   [3]     [2, 2]      [True]
```

# More Detail

Einsum Tuple language is an extension of Einstein Summation (einsum) notation, with these rules.

1. indices are tuples of unspecified length
2. tensors can be indexed with arbitrary expressions of indices (*index expressions*)
3. out-of-bounds index values are silently ignored
4. like einsum notation, broadcasting and summation are automatic
5. unlike einsum notation, indexing expressions appear in brackets, not subscripts



For quick examples of popular tensor operations defined in Einsum Tuple, see
the [ops directory](https://github.com/hrbigelow/einsum-tuple/tree/master/ops).
For the main documentation, see this
[article](https://www.mlcrumbs.com/einsum-tuple/public/index.html).

