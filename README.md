
# Einsum Tuple - a mini-language for defining tensor operations

## Motivation

Much mental effort in building ML models is spent keeping track of *tensor
shapes*, and how individual dimensions transform from inputs to outputs.  Some
dimensions match one-to-one between two inputs, or between an input and and
output.  Other dimensions may be simple functions of input dimensions.  Good
understanding of tensor operations requires knowing the relationships between
all of the dimensions of each tensor involved.

```python
# Tensor shapes
input.shape  = [10, 28, 28, 3]
filter.shape = [3, 3, 3, 8]
output.shape = [10, 26, 26, 8]

          |
          |
         \ /
          -

# Tensor signatures (template)
input signature  = [batch, input_pos, input_channel]
filter signature = [filter_pos, input_channel, output_channel]
output signature = [batch, output_pos, output_channel]

# named index groups (instantiation)
batch.shape          = [10]
input_pos.shape      = [28, 28]
filter_pos.shape     = [3, 3]
output_pos.shape     = input_pos.shape - filter_pos.shape + 1  # (broadcasted)
input_channel.shape  = [3]
output_channel.shape = [8]
```

In this repo, I introduce an intermediate concept of the *named index group* to
simplify this dimension-tracking problem.  Instead of viewing dimensions as
attributes of tensors directly, let tensors have a *signature*, an ordered list
of the named index groups.  Then, let each group have a *shape*.  Using this
two-step definition, tensor shapes are well defined yet much easier to mentally
track.  This also introduces a form of referential integrity \- if two tensors
share the same index group in their signatures, this is a declarative way to
establish that the sub-shapes must match.

With this new notation, many (not all) tensor operations may be defined
mathematically using an adapted form of Einstein summation notation.  This
adapted form is already used in official TensorFlow documentation, for example
[convolution](https://www.tensorflow.org/api_docs/python/tf/nn/convolution),
[matmul](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul),
[depthwise
convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d),
[dilation 2d](https://www.tensorflow.org/api_docs/python/tf/nn/dilation2d).

However, quite often, the formulas given there don't use meaningful index
names, even when meaningful names are used elsewhere in the same documentation
to define tensor shapes.  This repo introduces a syntax that merges both ideas
into a declarative, but executable formula to define many tensor operations.

One additional benefit to this notation is that, by giving names to *groups* of
dimensions rather than individual dimensions, one attains a level of genericity
to changing numbers of dimensions.  Thus, the same tensor signature and
expressions may be used for multiple instantiations of the operation.  For
example, convolution instantiated for 1, 2, or 3 spatial dimensions.

In this repo, the *named index group* concept is called an *Einsum Tuple* or
an *eintup*, for short.

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

This README shows some quick code examples with short explanations. 
For a more gradual introduction, see
[intro.md](https://github.com/hrbigelow/einsum-tuple/blob/master/intro.md).
The complete set of available examples of popular tensor operations defined in
Einsum Tuple are in [ops
directory](https://github.com/hrbigelow/einsum-tuple/tree/master/ops).  For a
technical description, see
[eintup.md](https://github.com/hrbigelow/einsum-tuple/blob/master/eintup.md)
and the formal grammar
[eintup_grammar.ebnf](https://github.com/hrbigelow/einsum-tuple/blob/master/eintup_grammar.ebnf).

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

# Constraints - DIMS constraints are necessary to avoid OOM errors
# The constraint on DIMS(opos) is needed for correctness.
RANK(ipos) IN [1,3]
DIMS(stride) IN [1,3]
DIMS(fpos) IN [2,5]
DIMS(ipos) IN [15,24]
DIMS(batch) IN [1,4]
DIMS(ichan) IN [1,3]
DIMS(ochan) IN [1,3]
DIMS(opos) = (DIMS(ipos) - DIMS(fpos) + 1) //^ DIMS(stride)
```

To validate the Einsum Tuple definition for convolution against
`tf.nn.convolution`, run the `.et` file.  The runtime system instantiates the
*Einsum Tuple* program with all valid combinations of ranks of Eintups, and
runs each instance.  The runtime instantiated values for `DIMS(batch)`,
`DIMS(ipos)` etc are shown in each output line.  In this example, the first
lines are 1D convolutions, followed by 2D and then 3D.

```bash
# Validate the above expression for 1D, 2D and 3D
# An optional number of reps can be specified as second argument (here, 5 reps)
y@henry-gs65:einsum-tuple$ python eintup.py ops/conv_valid_v1.et 5
batch   ipos           ichan   fpos        ochan   opos          stride         Valid
[3]     [18]           [3]     [4]         [1]     [8]           [2]            [True]
[2]     [24]           [1]     [5]         [3]     [10]          [2]            [True]
[1]     [23]           [2]     [4]         [2]     [10]          [2]            [True]
[4]     [15]           [3]     [3]         [1]     [5]           [3]            [True]
[2]     [20]           [2]     [2]         [2]     [19]          [1]            [True]
[2]     [17, 23]       [2]     [5, 3]      [3]     [5, 11]       [3, 2]         [True]
[4]     [23, 21]       [1]     [3, 2]      [1]     [7, 20]       [3, 1]         [True]
[2]     [18, 22]       [1]     [4, 3]      [3]     [8, 10]       [2, 2]         [True]
[2]     [21, 17]       [3]     [2, 3]      [3]     [20, 8]       [1, 2]         [True]
[2]     [23, 23]       [1]     [2, 2]      [1]     [8, 22]       [3, 1]         [True]
[4]     [15, 19, 23]   [1]     [4, 5, 2]   [2]     [12, 5, 22]   [1, 3, 1]      [True]
[3]     [20, 24, 18]   [1]     [5, 4, 5]   [2]     [6, 7, 5]     [3, 3, 3]      [True]
[2]     [20, 23, 17]   [1]     [4, 4, 5]   [1]     [17, 7, 5]    [1, 3, 3]      [True]
[4]     [15, 23, 19]   [3]     [3, 2, 3]   [1]     [13, 11, 9]   [1, 2, 2]      [True]
[2]     [22, 17, 21]   [2]     [5, 5, 5]   [2]     [18, 7, 17]   [1, 2, 1]      [True]
```

Note that the input dimensions are quite small.  This is necessary to avoid
out-of-memory errors, because the actual implementation unrolls the input or
filter into a higher dimensional structure.  The unrolled filter is
`filters[ipos-DIMS(stride)*opos,ichan,ochan]` and is
`RANK(ipos,opos,ichan,ochan) = 8` for 3D.  The unrolled input is
`input[batch,fpos+DIMS(stride)*opos,ichan]` and has
`RANK(batch,fpos,opos,ichan) = 8` as well, for the 3D case.  More detail about
rank and dimension calculation for expressions in the **Einsum Tuple** language
can be found in [intro.md](https://github.com/hrbigelow/einsum-tuple/blob/master/intro.md)

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

The Gather operation is unusual in that it employs tensor `indices` whose
values are used as indices into another tensor `params`.  It is also highly
polymorphic in the number of combinations of dimensions that it accepts. 
Here there are many possible valid rank combinations for the Eintups `batch`,
`elem`, `readloc` and `writeloc`.

A more gradual introduction and detailed explanation of `tf.gather_nd` is given
in [intro.md](https://github.com/hrbigelow/einsum-tuple/blob/master/intro.md).

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
...
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

See
[ops/flatten.et](https://github.com/hrbigelow/einsum-tuple/blob/master/ops/flatten.et)
for a simple example of `FLAT()` usage.

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

`DIMS(bsz)` provides the runtime constant `block_size`.  The index expression
`ipos//DIMS(bsz)` calculates the block that the `ipos` belongs to.  The
sub-expression `ipos % DIMS(bsz)` provides the position within the block, and
the full expression `FLAT(ipos % DIMS(bsz), ichan)` calculates the depth, as
described in the TensorFlow documentation for space_to_depth:

> Non-overlapping blocks of size block_size x block_size are rearranged into
> depth at each location.  The Y, X coordinates within each block of the input
> become the high order component of the output channel index.
> The depth of the output tensor is block_size * block_size * input_depth.

# More Detail

Einsum Tuple language is an extension of Einstein Summation (einsum) notation,
with these rules.

1. indices are tuples of unspecified length
2. tensors can be indexed with arbitrary expressions of indices (*index expressions*)
3. out-of-bounds index values are silently ignored
4. like einsum notation, broadcasting and summation are automatic
5. unlike einsum notation, indexing expressions appear in brackets, not subscripts

For quick examples of popular tensor operations defined in Einsum Tuple, see
the [ops directory](https://github.com/hrbigelow/einsum-tuple/tree/master/ops).
For a more gradual introduction, see
[intro.md](https://github.com/hrbigelow/einsum-tuple/blob/master/intro.md).
For a technical description, see
[eintup.md](https://github.com/hrbigelow/einsum-tuple/blob/master/eintup.md)
and the formal grammar
[eintup_grammar.ebnf](https://github.com/hrbigelow/einsum-tuple/blob/master/eintup_grammar.ebnf).


