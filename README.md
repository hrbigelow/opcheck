
# opschema 

A system to build input constraint schemas for TensorFlow operations

Install from PyPI:

    pip install opschema

## Motivation

TensorFlow Python ops give cryptic error messages. Often the
exceptions arise from several stack levels down the TensorFlow codebase.
Because of this, it is frequently not clear to the user what input constraints
are violated and what should be done to correct the error.

This is particularly challenging for ops that are highly polymorphic in the
combinations of shapes, data layouts and dtypes they accept. Documentation
often does not fully describe the legal inputs to ops. Finding out whether a
particular call is legal must be done by trial and error in many cases.

# Introduction

opschema provides an API for building *op schemas* for representing TensorFlow
operations.  Once written, a schema represents a single operation, such as
`tf.nn.convoution` or `tf.nn.bias_add`, etc.  The schema defines what inputs are
legal for the op.  Once defined, it provides three functionalities:

* wrap TensorFlow op, intercept inputs at call-time, provide human-readable error message 

* generate a complete set of legal (and a particular set of illegal) inputs for
  the op

* provide mathematically precise documentation of legal call
  configurations

* empirically validate schema correctness against TensorFlow
  op, given in TP, TN, FP and FN counts

## Synopsis

`opschema` offers serves as a registry for the available schemas and allows you
to load them individually or all together.  Schemas are instances of
`opschema.schema.OpSchema`, which provides member functions to configure it.
The schema definitions are in `opschema/ops`.

To see the list of implemented schemas, use:

```python
import opschema
ops = opschema.list_schemas()
print('\n'.join(op for op in ops))
tf.gather_nd
tf.nn.atrous_conv2d
tf.nn.atrous_conv2d_transpose
tf.nn.avg_pool
tf.nn.bias_add
tf.nn.convolution
tf.nn.depth_to_space
...
```

To print a human-readable representation of a schema, use:

```python
opschema.explain('tf.gather_nd', include_inventory=False)
```

Wrap the original TensorFlow op so that it opschema can intercept its inputs
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

    python -m opschema.cl validate <op_path> <reports_dir> [id_list]
    # example
    python -m opschema.cl validate tf.nn.convolution reports

The example produces files `reports/tf.nn.convolution.txt` and
`reports/tf.nn.convolution.sum.txt`.  If id_list is given, there will be one
entry for each id.  Otherwise, there is one entry for each input produced by
`generate_args()`.  

## How does it work?

To see a schema, run:

    python -m opschema.cl explain <op_path> [--include_inventory]
    # examples 
    python -m opschema.cl explain tf.nn.convolution

This provides a report in several sections, gradually explained below.

`opschema` uses three abstractions to define the schema:  *index*, *signature*,
and *layout*.

### Index

The lowest level abstraction is the *index*, created with the `OpSchema` API
function [add_index](opschema/schema.py:100)  This is a group of semantically
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

    python -c 'import opschema; opschema.explain("tf.nn.convolution")

To see the complete list of possible instantiations, use:

    python -c 'import opschema; opschema.explain("tf.nn.convolution", include_inventory=True)'




