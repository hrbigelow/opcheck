
# OpCheck - Better error messages for Tensor operations

##Synopsis

```python
import tensorflow as tf
import opcheck
opcheck.init()

# call tensorflow operations - receive better error messages
```

OpCheck is a wrapper around framework tensor operations which analyzes their
inputs.  If any inconsistency is found which would lead to an exception from
the framework, OpCheck will print an error message that is easier to read.
Otherwise, it is silent, and simply passes the inputs through to the wrapped
operation.

## How does it work?

OpCheck understands tensor operations in terms of three lower level concepts.
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
bkii   ffkl     ss       NCW          dd         bloo
bkiii  fffkl    sss      NCW          ddd        blooo
bik    fkl      s        NWC          d          bol
biik   ffkl     ss       NWC          dd         bool
biiik  fffkl    sss      NWC          ddd        boool
bki    fkl      s        NCHW         d          blo
bkii   ffkl     ss       NCHW         dd         bloo
bkiii  fffkl    sss      NCHW         ddd        blooo
bik    fkl      s        NHWC         d          bol
biik   ffkl     ss       NHWC         dd         bool
biiik  fffkl    sss      NHWC         ddd        boool
bki    fkl      s        NCDHW        d          blo
bkii   ffkl     ss       NCDHW        dd         bloo
bkiii  fffkl    sss      NCDHW        ddd        blooo
bik    fkl      s        NDHWC        d          bol
biik   ffkl     ss       NDHWC        dd         bool
biiik  fffkl    sss      NDHWC        ddd        boool
```

The explaination contains two sections.  The first section is a table listing
all of the OpCheck 'indices', with a one-letter code and its longer
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



