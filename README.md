
# OpSchema - Better error messages for Tensor operations

## Synopsis

```python
import opcheck
ops = opcheck.available_ops()
print('\n'.join(op for op in ops))
tf.gather_nd
tf.nn.atrous_conv2d
tf.nn.atrous_conv2d_transpose
tf.nn.avg_pool
tf.nn.bias_add
tf.nn.convolution
tf.nn.depth_to_space
tf.nn.separable_conv2d
tf.nn.space_to_batch
tf.nn.space_to_depth
tf.scatter_nd
...


# Wrap framework functions to be checked by OpSchema
opcheck.register('tf.nn.convolution', 'tf.gather_nd')
# or opcheck.register() to load all available ops

# Prints a symbolic list of all legal input tensor shapes, layouts, and dtypes
opcheck.explain('tf.nn.convolution')
...

# restore to original, unchecked framework function
opcheck.deregister('tf.nn.convolution')
```

OpSchema is a wrapper around framework tensor operations which analyzes their
inputs.  If any inconsistency is found which would lead to an exception from
the framework, OpSchema will print an error message that is easier to read.
Otherwise, it is silent, and simply passes the inputs through to the wrapped
operation.


## How does it work?

OpSchema understands tensor operations in terms of three lower level concepts.
The lowest level is the notion of an 'index', which is a group of semantically
related dimensions or other size-like quantities.  For example:

```bash
python explain.py tf.nn.convolution
Index  Description
b      batch
i      input spatial
f      filter spatial
o      output spatial
k      input channel
l      output channel
s      strides
d      dilations

input  filters  strides  data_format  dilations  return[0]
bki    fkl      s        NCW          d          blo
bik    fkl      s        NWC          d          bol
bkii   ffkl     ss       NCHW         dd         bloo
biik   ffkl     ss       NHWC         dd         bool
bkiii  fffkl    sss      NCDHW        ddd        blooo
biiik  fffkl    sss      NDHWC        ddd        boool
```

The explaination contains two sections.  The first section is a table listing
all of the OpSchema 'indices', with a one-letter code and its longer
description.  The second section shows a subset of shape-related arguments to
the operation.  Each line shows a combination of valid 'signatures' for these
arguments - for example, the first signature given for 'input' is 'bki'.  This
means a rank-3 input tensor with a batch dimension, then an input channel, then
an input spatial dimension.  

The one letter codes may be repeated if the individual dimensions belong to the
same semantic group.  For example, further down we see an 'input' signature of
'bkiii', which means it has three input spatial dimensions; this is a 3D
convolution.  The total set of lines in this section list all of the valid
geometries for this operation.

The `data_format` argument, if it exists, is part of the shape-related
arguments because it controls which combination of signatures is valid.



