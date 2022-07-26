
# Einsum Tuple - a mini-language for defining tensor operations

## Introduction

This repo defines and implements a runtime for **Einsum Tuple**, a high level
mini-language for defining tensor operations.  Each definition is
self-contained in an `.et` (Einsum Tuple) file.  Once defined, the Einsum Tuple
operation definition is evaluated using multiple combinations of shapes, and
verified against a provided framework call.  For example:

```
python eintup.py ops/conv_valid_v1.et
```

The central feature of the language is to assign meaningful names to *groups of
dimensions* in the input and output tensors.  Once named, the sizes of the
groups are subject to certain constraints which will be enumerated by the
system.  The constraints also serve as documentation for the user for valid
input combinations.

By naming groups of dimensions and allowing them to be variable in number, the
user can easily see which groups travel together and in what relative
positions, in each tensor.

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
params[batch,readloc,elem] = RANDOM(0, 10, FLOAT)
indices[batch,writeloc,coord] = RANDOM(0, DIMS(readloc)[coord], INT)
result[batch,writeloc,elem] = params[batch,indices[batch,writeloc,:],elem]


# Rank constraints
RANK(batch) IN [0,4]
RANK(readloc) IN [1,4]
RANK(writeloc) IN [1,3]
RANK(elem) IN [0,3]

# Validation
tf.gather_nd(params, indices, batch_dims=RANK(batch))
```

Here there are many possible valid rank combinations for the Eintups `batch`,
`elem`, `readloc` and `writeloc`.

```python
$ python eintup.py ops/gather_nd.et
batch       readloc       elem         writeloc      coord      Valid
[]          [6]           []           [2]           [1]        [True]
[]          [7]           []           [8, 1]        [1]        [True]
[]          [3]           []           [4, 5, 1]     [1]        [True]
[]          [3]           [4]          [5]           [1]        [True]
[]          [8]           [5]          [7, 1]        [1]        [True]
[]          [7]           [9]          [2, 7, 8]     [1]        [True]
[]          [10]          [6, 2]       [4]           [1]        [True]
[]          [7]           [10, 4]      [2, 4]        [1]        [True]
[]          [10]          [4, 7]       [4, 1, 5]     [1]        [True]
[]          [7]           [3, 6, 1]    [9]           [1]        [True]
[]          [4]           [3, 8, 6]    [9, 4]        [1]        [True]
[]          [9]           [8, 9, 4]    [10, 1, 8]    [1]        [True]
[]          [5, 2]        []           [2]           [2]        [True]
[]          [2, 7]        []           [6, 2]        [2]        [True]
[]          [1, 9]        []           [3, 3, 6]     [2]        [True]
[]          [3, 4]        [5]          [4]           [2]        [True]
[]          [4, 3]        [1]          [7, 2]        [2]        [True]
[]          [8, 4]        [6]          [10, 1, 4]    [2]        [True]
[]          [1, 1]        [9, 8]       [7]           [2]        [True]
[]          [6, 2]        [8, 6]       [1, 6]        [2]        [True]
[]          [1, 10]       [5, 5]       [6, 6, 7]     [2]        [True]
[]          [7, 8]        [3, 5, 5]    [3]           [2]        [True]
[]          [6, 6]        [7, 3, 9]    [4, 7]        [2]        [True]
[]          [3, 7]        [5, 7, 10]   [4, 10, 5]    [2]        [True]
[]          [7, 3, 3]     []           [1]           [3]        [True]
[]          [7, 9, 5]     []           [8, 6]        [3]        [True]
[]          [8, 1, 4]     []           [1, 9, 4]     [3]        [True]
[]          [4, 2, 8]     [1]          [9]           [3]        [True]
[]          [5, 6, 10]    [6]          [9, 9]        [3]        [True]
[]          [8, 1, 4]     [8]          [6, 9, 9]     [3]        [True]
[]          [8, 4, 1]     [4, 8]       [9]           [3]        [True]
[]          [6, 1, 10]    [10, 5]      [4, 1]        [3]        [True]
[]          [1, 5, 10]    [9, 6]       [2, 10, 5]    [3]        [True]
[]          [4, 3, 10]    [10, 3, 6]   [9]           [3]        [True]
[]          [5, 1, 9]     [1, 7, 3]    [3, 3]        [3]        [True]
[]          [2, 3, 5]     [3, 3, 4]    [4, 10, 2]    [3]        [True]
[3]         [9]           []           [2]           [1]        [True]
[4]         [3]           []           [2, 5]        [1]        [True]
[0]         [6]           []           [3, 4, 9]     [1]        [True]
[2]         [9]           [1]          [9]           [1]        [True]
[3]         [1]           [1]          [2, 5]        [1]        [True]
[0]         [3]           [1]          [4, 6, 2]     [1]        [True]
[2]         [9]           [7, 1]       [3]           [1]        [True]
[0]         [4]           [9, 1]       [4, 2]        [1]        [True]
[3]         [7]           [6, 3]       [8, 2, 2]     [1]        [True]
[2]         [2]           [6, 4, 1]    [8]           [1]        [True]
[2]         [3]           [4, 6, 1]    [7, 1]        [1]        [True]
[0]         [5]           [4, 1, 2]    [8, 2, 6]     [1]        [True]
[0]         [9, 8]        []           [6]           [2]        [True]
[1]         [8, 5]        []           [5, 8]        [2]        [True]
[0]         [3, 1]        []           [4, 8, 2]     [2]        [True]
[4]         [9, 9]        [3]          [7]           [2]        [True]
[1]         [5, 4]        [5]          [2, 1]        [2]        [True]
[4]         [7, 9]        [10]         [6, 6, 3]     [2]        [True]
[3]         [10, 10]      [6, 7]       [1]           [2]        [True]
[3]         [6, 8]        [4, 10]      [9, 6]        [2]        [True]
[3]         [2, 8]        [8, 8]       [1, 1, 3]     [2]        [True]
[1]         [3, 10]       [6, 9, 4]    [5]           [2]        [True]
[2]         [8, 3]        [7, 10, 1]   [5, 8]        [2]        [True]
[2]         [8, 6]        [8, 10, 2]   [2, 4, 5]     [2]        [True]
[1]         [3, 3, 5]     []           [5]           [3]        [True]
[2]         [3, 2, 3]     []           [6, 8]        [3]        [True]
[4]         [4, 5, 1]     []           [6, 8, 10]    [3]        [True]
[4]         [1, 3, 5]     [1]          [5]           [3]        [True]
[2]         [4, 5, 6]     [10]         [4, 10]       [3]        [True]
[1]         [7, 10, 5]    [7]          [9, 10, 2]    [3]        [True]
[1]         [2, 1, 4]     [6, 9]       [1]           [3]        [True]
[1]         [8, 7, 4]     [2, 7]       [6, 4]        [3]        [True]
[2]         [2, 10, 10]   [6, 4]       [3, 3, 2]     [3]        [True]
[2]         [7, 3, 9]     [1, 5, 5]    [6]           [3]        [True]
[3]         [3, 10, 1]    [5, 9, 10]   [5, 2]        [3]        [True]
[0]         [3, 5, 7]     [4, 3, 5]    [4, 6, 8]     [3]        [True]
[1, 1]      [8]           []           [9]           [1]        [True]
[1, 3]      [2]           []           [6, 10]       [1]        [True]
[2, 2]      [9]           []           [7, 8, 7]     [1]        [True]
[2, 0]      [1]           [10]         [6]           [1]        [True]
[1, 0]      [6]           [2]          [8, 6]        [1]        [True]
[4, 1]      [6]           [10]         [8, 2, 3]     [1]        [True]
[2, 0]      [1]           [9, 10]      [5]           [1]        [True]
[1, 4]      [5]           [1, 9]       [4, 8]        [1]        [True]
[4, 4]      [3]           [2, 8]       [6, 5, 4]     [1]        [True]
[3, 3]      [1]           [3, 7, 3]    [6]           [1]        [True]
[1, 3]      [6]           [5, 2, 2]    [7, 1]        [1]        [True]
[2, 4]      [5]           [7, 8, 6]    [6, 6, 7]     [1]        [True]
[1, 4]      [7, 6]        []           [3]           [2]        [True]
[2, 1]      [5, 2]        []           [1, 8]        [2]        [True]
[0, 4]      [10, 3]       []           [1, 10, 10]   [2]        [True]
[4, 0]      [10, 9]       [8]          [5]           [2]        [True]
[0, 1]      [5, 2]        [4]          [1, 1]        [2]        [True]
[0, 2]      [8, 9]        [8]          [7, 2, 4]     [2]        [True]
[0, 3]      [1, 5]        [7, 8]       [9]           [2]        [True]
[0, 1]      [9, 1]        [8, 9]       [7, 3]        [2]        [True]
[2, 2]      [3, 3]        [6, 4]       [10, 5, 10]   [2]        [True]
[2, 1]      [3, 10]       [8, 9, 3]    [7]           [2]        [True]
[3, 3]      [9, 3]        [3, 1, 10]   [5, 9]        [2]        [True]
[3, 4]      [3, 2]        [3, 1, 6]    [4, 8, 8]     [2]        [True]
[0, 1]      [9, 8, 6]     []           [7]           [3]        [True]
[1, 4]      [2, 3, 7]     []           [1, 2]        [3]        [True]
[0, 4]      [7, 1, 2]     []           [3, 9, 6]     [3]        [True]
[0, 3]      [5, 3, 8]     [1]          [1]           [3]        [True]
[3, 0]      [3, 9, 6]     [3]          [5, 9]        [3]        [True]
[1, 2]      [2, 5, 4]     [10]         [9, 5, 7]     [3]        [True]
[1, 1]      [7, 6, 8]     [3, 5]       [6]           [3]        [True]
[2, 2]      [1, 5, 10]    [9, 3]       [9, 4]        [3]        [True]
[4, 1]      [5, 1, 10]    [1, 10]      [1, 7, 9]     [3]        [True]
[3, 2]      [4, 7, 8]     [1, 3, 4]    [9]           [3]        [True]
[3, 1]      [4, 9, 4]     [1, 3, 9]    [10, 3]       [3]        [True]
[2, 4]      [8, 10, 4]    [8, 2, 4]    [9, 6, 2]     [3]        [True]
[4, 3, 0]   [1]           []           [9]           [1]        [True]
[2, 2, 2]   [8]           []           [9, 6]        [1]        [True]
[1, 0, 4]   [10]          []           [9, 5, 8]     [1]        [True]
[1, 3, 3]   [7]           [7]          [5]           [1]        [True]
[4, 0, 1]   [2]           [1]          [6, 2]        [1]        [True]
[3, 0, 2]   [3]           [7]          [2, 5, 10]    [1]        [True]
[1, 3, 4]   [7]           [7, 9]       [8]           [1]        [True]
[4, 0, 3]   [2]           [4, 9]       [5, 9]        [1]        [True]
[4, 0, 0]   [7]           [4, 5]       [8, 10, 7]    [1]        [True]
[1, 2, 0]   [4]           [10, 7, 2]   [7]           [1]        [True]
[3, 1, 0]   [3]           [6, 6, 8]    [5, 8]        [1]        [True]
[2, 4, 4]   [4]           [9, 10, 7]   [6, 4, 1]     [1]        [True]
[1, 0, 2]   [6, 4]        []           [3]           [2]        [True]
[3, 3, 4]   [10, 7]       []           [3, 8]        [2]        [True]
[0, 2, 3]   [6, 4]        []           [9, 10, 6]    [2]        [True]
[0, 2, 4]   [5, 4]        [9]          [6]           [2]        [True]
[4, 3, 0]   [5, 8]        [8]          [6, 5]        [2]        [True]
[2, 4, 3]   [8, 1]        [9]          [5, 9, 6]     [2]        [True]
[1, 3, 2]   [6, 7]        [8, 6]       [6]           [2]        [True]
[4, 0, 1]   [8, 1]        [4, 7]       [4, 7]        [2]        [True]
[4, 1, 2]   [5, 1]        [3, 8]       [6, 1, 6]     [2]        [True]
[1, 1, 2]   [3, 1]        [8, 6, 9]    [4]           [2]        [True]
[2, 0, 1]   [1, 8]        [2, 8, 10]   [3, 1]        [2]        [True]
[4, 4, 3]   [5, 10]       [8, 1, 3]    [6, 7, 8]     [2]        [True]
[4, 2, 0]   [10, 8, 7]    []           [7]           [3]        [True]
[0, 1, 4]   [5, 5, 10]    []           [7, 3]        [3]        [True]
[0, 2, 3]   [8, 1, 8]     []           [6, 1, 6]     [3]        [True]
[1, 3, 3]   [5, 2, 9]     [8]          [6]           [3]        [True]
[0, 3, 1]   [4, 2, 6]     [5]          [7, 2]        [3]        [True]
[2, 1, 3]   [1, 4, 10]    [3]          [9, 4, 9]     [3]        [True]
[2, 0, 4]   [9, 4, 6]     [5, 10]      [5]           [3]        [True]
[2, 4, 3]   [9, 1, 1]     [4, 8]       [3, 4]        [3]        [True]
[3, 2, 1]   [8, 8, 9]     [4, 1]       [5, 3, 4]     [3]        [True]
[1, 3, 3]   [9, 7, 9]     [2, 1, 10]   [9]           [3]        [True]
[2, 4, 1]   [3, 1, 6]     [3, 10, 1]   [4, 2]        [3]        [True]
[0, 4, 2]   [9, 5, 6]     [2, 8, 6]    [8, 6, 7]     [3]        [True]
```

### Meshgrid

Einsum Tuple allows a straightforward expression of broadcasting behavior.  Any
Eintups which appear on the left hand side but not the right are automatically
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

# Validate against tf meshgrid call.  The call returns a tuple of four tensors
# which are validated against out1, out2, out3, and out4
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
For a more gradual introduction, see
[intro.md](https://github.com/hrbigelow/einsum-tuple/blob/master/intro.md)

