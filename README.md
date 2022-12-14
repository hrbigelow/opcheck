
# opschema 

A system to build input constraint schemas for TensorFlow operations

Install from PyPI:

    pip install opschema

# Motivation

TensorFlow Python is a workhorse of the Machine Learning world used by many
thousands of developers.  However, as an API, it is challenging.  Tensor ops
are often highly polymorphic with intricate shape and other required
relationships in inputs.  If these are not met, often the exception will arise
from several stack levels down the codebase.  Because of this, it is
frequently not clear to the user what input constraints are violated and what
should be done to correct the error.

Documentation very often does not fully describe the legal inputs to ops. Finding
out whether a particular call is legal must be done by trial and error in many
cases.

In some cases, the API requires redundant information to be provided.  For
example,
[tf.nn.atrous_conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d_transpose)
and
[tf.nn.conv_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv_transpose)
require an `output_shape` parameter which requires the user to restate the
'batch' and 'out_channel' dimensions, and compute the out_height and out_width
manually.  This is also the case with

Many ops accept a `data_format` parameter which takes on values such as 'NCW',
'NCHW', 'NCDHW', 'NWC', 'NHWC' and 'NDHWC'.  This parameter is really
communicating the notion of a *layout* which is either *channel first* or
*channel last*.  Which variety of `data_format` is needed is already
communicated by the `filter` shape.  

In fact, contraray to documentation, 
[tf.nn.convolution](https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
actually does accept 'NWC', 'NCW' values for `data_format` for some 2D
convolutions.

# Introduction

opschema provides an API for building *op schemas* for representing TensorFlow
operations.  Once written, a schema represents a single operation, such as
`tf.nn.convoution` or `tf.nn.bias_add`, etc.  The schema defines what inputs are
legal for the op.  Once defined, it provides four functionalities:

* wrap TensorFlow op, intercept inputs at call-time, provide human-readable
  error message 

* generate a complete set of legal (and a particular set of illegal) inputs for
  the op

* provide mathematically precise documentation of legal call
  configurations

* empirically validate schema correctness against TensorFlow
  op, given in TP, TN, FP and FN counts

## Synopsis

`opschema` provides a registry for the available schemas and allows you
to load them individually or all together.  Schemas are instances of
`opschema.schema.OpSchema`, which provides member functions to configure it.
The schema definitions are in `opschema/ops`.

To see the list of implemented schemas, use:

    python -m opschema.cl list

To print a human-readable representation of a schema, use one of:

    python -m opschema.cl explain tf.gather_nd
    python -m opschema.cl explain tf.gather_nd -i
    python -m opschema.cl explain tf.gather_nd --include_inventory

Note that including the inventory may be very long for highly polymorphic ops.

To wrap the original TensorFlow op so that it opschema can intercept its inputs
and provide error messages.  

```python
# wrap tf.gather_nd
opschema.register('tf.gather_nd')

# call tf.gather_nd(...) directly

# restore tf.gather_nd to original
opschema.deregister('tf.gather_nd')
```

This process reassigns the member function, for example `tf.gather_nd` to a
wrapper function.  The wrapper function first inspects the inputs and prints an
error message if any violation is detected.  Regardless of violation, it then
passes the inputs on to the original TensorFlow operation.  In this way it is
otherwise unobtrusive to the functioning of an existing network.

## Example Error messages - before and after

Run

    python -m opschema.cl validate OP_PATH REPORTS_DIR [TEST_IDS] [SKIP_IDS] [ERROR_QUOTA]
    # example
    python -m opschema.cl validate tf.nn.convolution reports

The example produces files `reports/tf.nn.convolution.txt` and
`reports/tf.nn.convolution.sum.txt`.  

## How does it work?

`opschema` uses three abstractions to define the schema:  *index*, *signature*,
and *layout*.

### Index

The lowest level abstraction is the *index*, created with the `OpSchema` API
function [add_index](opschema/schema.py#L877).  This is a group of semantically
related dimensions that occur within the shape of input tensors or other
shape-related arguments.  An index has a single-letter name and a longer
description.  It is rank-agnostic in that different calls to the op may take on
a different number of these dimensions.  The individual components of the
dimensions often participate in formulas with dimensions of other indices.

Examples:

    code  description
    b     batch
    i     input spatial
    k     input channel
    f     filter spatial
    j     output filter 
    l     output channel

Rank-agnostic here means that, at run-time, an index can represent zero, one,
two, or more individual dimensions within a tensor shape, depending on how the
op was called.

### Signature

A *signature* is simply an ordered sequence of *indexes*, usually represented
as a string of the one-letter codes.  Most input tensors have a *signature*.
Importantly, since each *index* is rank-agnostic, so is the signature.

Examples:

    tensor   signature
    input    bik           
    filter   fjl

While indexes are rank-agnostic, it is also useful to see possible
*instantiations* of indexes showing the actual rank of the shape for a
particular call of the op.  For instance, `tf.nn.convolution` may be called
with 1, 2, or 3 spatial dimensions, which imply the rank of indexes 'i' and
'f'.  Similarly, it works with any number of batch dimensions 'b' >= 1.  Such
instantiations can be represented using repetitions of the one-letter code:

Examples:

    input shape instantiations
    bik, biik, biiik, bbik, bbbik, ...

By default, each index has no constraints on what rank it can take on.
Rank constraints are provided within the [add_index](opschema/schema.py#L877)
API function.  The 'explain' command-line function has a section on the index
rank constraints.  For `tf.nn.convolution` it is:

```
Index ranks

rank(b) in [1, 5]
rank(i) in [1, 3]
rank(f) = rank(i)
rank(p) = rank(i)
rank(s) = rank(i)
rank(d) = rank(i)
rank(k) = 1
rank(j) = 1
rank(l) = 1
rank(o) = rank(i)
```

The rank of an index can be constrained either within a range, or constrained
to be equal to that of another index.  In the above, the rank of the 'input
spatial' (i) index can be in [1,3], and there can be between 1 and 5 batch
dimensions.  The rank of the 'filter spatial' (f) index is set equal to that of
'input spatial' and so on.

#### Computed Indexes and intermediate Indexes

Indexes come in two varieties:  *computed* or not computed.  Being *computed*
means that the dimensions are determined as a function of the dimensions of
other indices.  A function is assigned using the API call
[OpSchema.comp_dims_cw](opschema/schema.py#1130) for a component-wise computation, or
[OpSchema.comp_dims](opschema/schema.py#1113) for a non-component-wise computation.  For
example, with `tf.nn.convolution` the 'explain' section shows:

```bash
Computed dimensions

p = (f - 1) * d + 1
o = ceil((i - p + 1) / s)   [padding = VALID]
o = ceil(i / s)   [padding = SAME]
```

These are registered as component-wise.  For instance, if `rank(i) = 2`, then
each dimension of `i` will be computed from the corresponding dimension of `s`
and/or `p`, and so forth.

Here, we see that 'padded filter spatial' (p) index is computed from 'filter
spatial' and 'dilation'.  And, 'output spatial' (o) has two different formulas,
depending on the command-line argument 'padding'.

Note here that the 'p' index does not appear anywhere in an input signature.
It is purely an intermediate calculation.  But, having an explicit name for the
index is useful to clarify to the user how the visible 'o' index is computed in
the case of 'padding = VALID'.

These formulas are also used to display actual dimensions in error messages.

### Layout

A *layout* is a set of consistent *signatures* accepted by the op.  Some ops
have just a single layout.  May have two, which could be described as 'channel
first' or 'channel last', and are determined by the `data_format` argument.

Examples:

    input  filters  strides  dilations  return[0]  data_format
    bki    fjl      s        d          blo        ['NCW', 'NCHW', 'NCDHW']
    bik    fjl      s        d          bol        ['NWC', 'NHWC', 'NDHWC']

The above example shows two different layouts for the `tf.nn.convolution`
operation.  Like *signatures*, the notion of a *layout* is rank-agnostic.  

The indexes and layouts for a given op schema can be shown with:

    python -m opschema.cl explain tf.nn.convolution

To see the complete list of signature instantiations, use:

    python -m opschema.cl explain tf.nn.convolution -i

# DType constraints

TensorFlow ops are usually constrained to work on certain combinations of
`dtype` of the input tensors.  `opschema` provides a few API functions to
specify this.

```python
# DType constraints for tf.nn.convolution
op.valid_dtypes('input', ('int32', 'float', 'bfloat16'))
op.equate_dtypes('filters', 'input')
op.exclude_combos('input', 'int32', 'i', (1,2), LAYOUT, 0)
op.exclude_combos('input', 'int32', 'i', 3)
op.exclude_combos('input', 'bfloat16', 'i', (1,2))
op.exclude_combos('input', 'bfloat16', 'i', 3, LAYOUT, 0)
```

The above snippet of the `tf.nn.convolution` schema definition illustrates the
three `OpSchema` API calls related to dtype constraints.  `valid_dtypes` simply
specifies which dtypes are accepted for a given argument tensor.  There is a
wildcard-like syntax (see [base.py:parse_dtype_expr](opschema/base.py#L264))
used to specify multiple dtypes briefly.

`equate_dtypes` says that the dtype of one argument tensor must be identical to
another.

`exclude_combos` declares that a certain combination of dtypes, index ranks,
and/or layouts is excluded.  This is needed because certain of these
combinations may not be implemented by TensorFlow.

# Other Constraints

There are other relationships between inputs in certain TensorFlow ops.  For
example, with `tf.gather_nd`, the last dimension of the `indices` shape
determines the rank of the 'read location' (r) index.  This is declared using
the API function [OpSchema.rank_dims_constraint](opschema/schema.py#L1638).



