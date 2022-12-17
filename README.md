
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

* provide better error messages than the exceptions TensorFlow issues

* generate a complete set of legal (and a particular set of illegal) inputs for
  the op

* provide mathematically precise documentation of legal call
  configurations

* empirically validate schema correctness against TensorFlow
  op, given in TP, TN, FP and FN counts

## Synopsis

List available op schemas (defined under opschema/ops)

    python -m opschema.cl list

Explain an op, optionally including a list of all possible call configurations

    python -m opschema.cl explain OP_PATH [-i|--include_inventory]

Print the graphs associated with an op in .pdf format (requires graphviz)

    python -m opschema.cl graph OP_PATH OUT_DIR

Validate an op schema against the TensorFlow op it represents  

    python -m opschema.cl validate OP_PATH OUT_DIR \
        [--test_ids] \
        [--skip_ids] \
        [--max_dtype_err=0] \
        [--rand_seed=0] \
        [--show_traceback]

## What it does

`opschema` provides an API for writing *schemas* for TensorFlow ops.  A schema
here means a set of rules that define what combinations of inputs are legal.
Once a schema is defined, you can use opschema to generate a complete set of
test inputs for the op for all legal combinations of tensor dtypes, shapes, and
combinations of other control arguments such as `data_format` etc.  In
addition, a subset of illegal inputs can be generated as well, which are useful
for comparing TensorFlow's exception with opschema's error message.

## Example Error Messages

Some examples TensorFlow calls that raised exceptions.  Each example shows the
argument values (tensors are abbreviated to shape:dtype), the TensorFlow
exception text, and the error message from opschema.

Examples are generated with:

    python -m opschema.cl validate OP_PATH OUT_DIR

which produces files OUT_DIR/OP_PATH.txt and OUT_DIR/OP_PATH.sum.txt

These excerpts are taken from OUT_DIR/OP_PATH.txt.  The format for each entry
is:

    ## ID  CLASS  ARGS_LIST 
    TF_EXCEPTION_TEXT
   
    OPSCHEMA_ERROR_MESSAGE

CLASS has the following meaning: 

    CLASS     TensorFlow     opschema
    TP        raises         issues error
    TN        succeeds       none 
    FP        succeeds       issues error
    FN        raises         none 

Note that CLASS does not say anything about how well the TensorFlow exception
and opschema error message agree.  The goal is for the opschema message to be
more informative and lead to a successful correction.  But, the schema
definition is a reverse-engineering process based on the observed behavior of
the TensorFlow op. 

### `tf.nn.convolution`

```
## 2    TP      input=[42, 23, 5]:float16, filters=[32, 23, 24]:int8, strides=1, padding=VALID, data_format=NCW, dilations=1
cannot compute Conv2D as input #1(zero-based) was expected to be a half tensor but is a int8 tensor [Op:Conv2D]

Received filters.dtype = int8 and input.dtype = float16.  dtypes of filters and input must match.
```

Here, input and filters have mismatching dtypes.  TensorFlow's exception
assumes filters has the wrong one, which may be too strong.  Also, it refers to
filters as 'input #1(zero_based)'.  opschema always uses the actual names of
arguments in its error messages.

```
## 3    TP      input=[87, 80, 3]:float16, filters=[9, 2, 3]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
Computed output size would be negative: -5 [input_size: 3, effective_filter_size: 9, stride: 1] [Op:Conv2D]

           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received       87 80 3           9 2 3         1           NCW           1           87 3 -5
template        b  k i           f j l         s                         d            b l  o
   error                                                                                  ^^

output_spatial (o) = [-5].  output_spatial must be >= 0

Dimensions computed as:
dilated_filter_spatial = (filter_spatial - 1) * dilations + 1
output_spatial = ceil((input_spatial + dilated_filter_spatial - 1) / strides)   [padding = VALID]

g = (f - 1) * d + 1
o = ceil((i + g - 1) / s)   [padding = VALID]

[9] = ([9] - 1) * 1 + 1
[-5] = ceil(([3] + [9] - 1) / 1)   [padding = VALID]
```

Here, TensorFlow's error message is good.  Although, it seems the internal
name 'input_size' is not chosen in coordination with the documentation.  It is
also confusing that Op:Conv2D is mentioned, since this is a 1D convolution.

opschema's error message highlights exactly which indices are negative, and
shows the formulas used to compute them, using top-level names.  These same
names (input_spatial, output_spatial, dilated_filter_spatial, etc) appear in
the automatically generated docs as well.

```
## 5    TP      input=[87, 1, 3]:float16, filters=[9, 2, 3]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
input depth must be evenly divisible by filter depth: 1 vs 2 [Op:Conv2D]

           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received        87 1 3           9 2 3         1           NCW           1           87 3 -5
template         b k i           f j l         s                         d            b l  o
   error           ^               ^

input_channel (k) = [1] and filter_input_channel (j) = [2].  input_channel must be divisible by filter_input_channel
```

Here TensorFlow's error message is not bad.  Again it uses non-standard names
'input depth' and 'filter depth'.  'input depth' is equivalent to the
documentation's term 'in_channels'.  However, the documentation erroneously
claims that filters must have matching depth as input.

    filters	    An (N+2)-D Tensor with the same type as input and shape 
                spatial_filter_shape + [in_channels, out_channels].

opschema by design always uses the standard names and highlights the exact
dimensions in context.

```
## 4    FN      input=[87, 80, 3]:float16, filters=[1, 2, 3]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
No algorithm worked! [Op:Conv2D]

None
```

This is a case opschema thinks is fine, and I'm not sure what the problem is.
It seems to have something to do with the relationship between indices k
(input_channel) and j (filter_input_channel).

```
## 25   TP      input=[97, 27, 1]:float16, filters=[72, 1, 9, 20]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
Value for attr 'data_format' of "NCW" is not in the list of allowed values: "NHWC", "NCHW"
        ; NodeDef: {{node Conv2D}}; Op<name=Conv2D; signature=input:T, filter:T -> output:T; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32]; attr=strides:list(int); attr=use_cudnn_on_gpu:bool,default=true; attr=padding:string,allowed=["SAME", "VALID", "EXPLICIT"]; attr=explicit_paddings:list(int),default=[]; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]; attr=dilations:list(int),default=[1, 1, 1, 1]> [Op:Conv2D]

Received invalid configuration: input rank = 3, filters rank = 4 and data_format = NCW.  Closest valid configurations:

           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received     [97,27,1]     [72,1,9,20]         1           NCW           1
config 1         b k i        => f j l         s                         d             b l o

=> config 1: remove 1 dimension from filters
```

In this example, it could be intended as a 2D convolution but with a dimension
missing from input, or that filters has one extra dimension.  TensorFlow
assumes the latter, and this could cause confusion for the user.  Also, the
extra information doesn't seem helpful.

opschema is admittedly a bit didactic here - it tries to guess the closest fix
based on an internal 'edit distance'.


```
## 26   TP      input=[67, 36, 12]:float16, filters=[18, 21]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
num_spatial_dims (input.shape.ndims - num_batch_dims - 1) must be one of 1, 2 or 3 but saw 0.  num_batch_dims: 2.

Received invalid configuration: input rank = 3, filters rank = 2 and data_format = NCW.  Closest valid configurations:

           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received    [67,36,12]         [18,21]         1           NCW           1
config 1         b k i        => f j l         s                         d             b l o

=> config 1: add 1 dimension to filters
```

Here, TensorFlow's message claims `num_batch_dims: 2` which is too strong of an
assumption, possibly leading to confusion.

```
## 59   TP      input=[69, 45]:float32, filters=[59, 23, 2]:float32, strides=1, padding=VALID, data_format=NCW, dilations=1
input must be 4-dimensional[69,1,45] [Op:Conv2D]

Received invalid configuration: input rank = 2, filters rank = 3 and data_format = NCW.  Closest valid configurations:

           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received       [69,45]       [59,23,2]         1           NCW           1
config 1      => b k i           f j l         s                         d             b l o

=> config 1: add 1 dimension to input
```

Here TensorFlow's message is contradictory and nearly uninterpretable.  It
seems to claim that 'input' has a shape [69,1,45] which it does not actually
have.

```
## 98   FP      input=[83, 5, 90]:float32, filters=[90, 5, 25]:float32, strides=1, padding=VALID, data_format=NCHW, dilations=1
None

           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received       83 5 90         90 5 25         1          NCHW           1           83 25 1
template        b k  i          f j  l         s           NCW           d            b  l o
   error                                                  ^^^^

=> Change data_format to NCW
```

Here, TensorFlow succeeds, performing a 1D convolution successfully, even
though `data_format` was provided incorrectly as `NCHW`, contrary to the
documentation.

```
## 1868 FP      input=[99, 1, 90, 31]:float16, filters=[40, 15, 18]:float16, strides=[4], padding=SAME, data_format=NCW, dilations=1
None

Return tensor 0 was expected to have shape [99, 1, 18, 8] but was [99, 1, 18, 31]
```

Here is another false positive case in which TensorFlow does not raise an
exception.  opschema flags it because the return tensor didn't have the
expected shape.  


### `tf.nn.avg_pool`

```
## 1    TP      input=[66, 17, 1]:bfloat16, ksize=[7], strides=[50], padding=VALID, data_format=NCW
Tried to squeeze dim index 2 for tensor with 1 dimensions. [Op:Squeeze]

This combination is not implemented: input.dtype in (bfloat16) and [1] input_spatial dimensions
```

Here, TensorFlow's exception does not give the user any clue what went wrong.
opschema's error message in my opinion could be improved, but is reasonably
clear.  Perhaps it could be augmented with a template table as in previous
examples.

```
## 9    TP      input=[2, 36, 1]:float16, ksize=[5, 1], strides=[67], padding=VALID, data_format=NCW
ksize should be of length 1, 1 or 3 but was 2

Received invalid configuration: input rank = 3, ksize rank = 2 and data_format = NCW.  Closest valid configurations:

           input.shape   ksize   strides   data_format   return[0].shape
received      [2,36,1]   [5,1]        67           NCW
config 1         b c i    => k         s                           b c o

=> config 1: remove 1 dimension from ksize
```

TensorFlow guesses that the rank of ksize is the problem, but suggests '1, 1,
or 3', which doesn't make much sense.

```
## 13   TP      input=[91, 19, 5]:float16, ksize=[5], strides=[14, 1], padding=VALID, data_format=NCW
strides should be of length 1, 1 or 3 but was 2

Received invalid configuration: input rank = 3, strides rank = 2 and data_format = NCW.  Closest valid configurations:

           input.shape   ksize   strides   data_format   return[0].shape
received     [91,19,5]       5    [14,1]           NCW
config 1         b c i       k      => s                           b c o

=> config 1: remove 1 dimension from strides
```

Here TensorFlow's exception seems to have the same problem as in example 9.
Also, the documentation contains a similar confusing message about `ksize` and
`strides` parameters:

	  strides   An int or list of ints that has length 1, N or N+2. The stride of the 
              sliding window for each dimension of the input tensor.

It should read '1, 2, or 3'.

```
## 42   FP      input=[37, 40, 6]:float16, ksize=[8], strides=[98], padding=VALID, data_format=NCHW
None

           input.shape   ksize   strides   data_format   return[0].shape
received       37 40 6       8        98          NCHW           37 40 0
template        b  c i       k         s           NCW            b  c o
   error                                          ^^^^

=> Change data_format to NCW

           input.shape   ksize   strides   data_format   return[0].shape
received       37 40 6       8        98          NCHW            37 1 6
template        b  i c       k         s           NWC             b o c
   error                                          ^^^^

=> Change data_format to NWC

Received invalid configuration: input rank = 3 and data_format = NCHW.  Closest valid configurations:

           input.shape   ksize   strides   data_format   return[0].shape
received     [37,40,6]       8        98          NCHW
config 1    => b c i i       k         s                         b c o o

=> config 1: add 1 dimension to input
```

Here, as in `tf.nn.convolution` example 98, TensorFlow allows `data_format =
NCHW` in contradiction with documentation.  opschema is admittedly trying a bit
too hard and is too verbose (probably needs to be pared down).

```
## 51   TP      input=[63, 34, 2]:float16, ksize=[3], strides=[90], padding=VALID, data_format=NHWC
Can not squeeze dim[2], expected a dimension of 1, got 0 [Op:Squeeze]

<opschema response omitted>
```

TensorFlow's error message is not useful at all here.


### `tf.nn.depth_to_space`

```
## 17   TP      input=[90, 306, 1, 51]:int32, block_size=1, data_format=NHWC
Value for attr 'block_size' of 1 must be at least minimum 2
        ; NodeDef: {{node DepthToSpace}}; Op<name=DepthToSpace; signature=input:T -> output:T; attr=T:type; attr=block_size:int,min=2; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW", "NCHW_VECT_C"]> [Op:DepthToSpace]

Argument 'block_size' expected to be an integer >= 2
```

The first part of this message is not bad, except that it is confusing to call
it attr 'block_size'.  And, the remaining part is just visual noise.


### `tf.nn.space_to_batch`

```
## 7    TP      input=[40, 36]:int8, block_shape=[31, 1], paddings=[11], [15]
input rank should be >= 3 instead of 2 [Op:SpaceToBatchND]

Received invalid configuration: input rank = 2, block_shape rank = 2, paddings.0 rank = 1 and paddings.1 rank = 1.  Closest valid configurations:

           input.shape   block_shape   return[0].shape
received       [40,36]        [31,1]
config 1           b i          => k               p o

=> config 1: remove 1 dimension from block_shape
```

Comparing the received signatures with closest valid signatures:

                input  block_shape  paddings.0  paddings.1
    received    bi     kk           s           e
    config 1    bi     k            s           e
    config 2    bii    kk           ss          ee

opschema suggests the closest valid one (config 1) which implies that
block_shape has one too many dimensions.  The other possibility is that it was
correct, but that both input and paddings are missing a dimension.

TensorFlow assumes the latter, which may be too strong of an assumption.

```
## 8    TP      input=[47, 34]:int8, block_shape=[], paddings=[1], [23]
paddings should have shape [0, 2] instead of [1,2] [Op:SpaceToBatchND]

Received invalid configuration: input rank = 2, block_shape rank = 0, paddings.0 rank = 1 and paddings.1 rank = 1.  Closest valid configurations:

           input.shape   block_shape   return[0].shape
received       [47,34]            []
config 1           b i          => k               p o

=> config 1: add 1 dimension to block_shape
```

Here, 'block_shape' was incorrectly provided with rank 0, when it must be
between 1 and 3.  But, TensorFlow incorrectly suggests to change the 'paddings'
parameter rank to match that of 'block_shape', leading to a nonsensical
suggestion.


```
## 21   TP      input=[1, 17, 4]:int16, block_shape=[47], paddings=[34], [9]
padded_shape[0]=60 is not divisible by block_shape[0]=47 [Op:SpaceToBatchND]

           input.shape   block_shape   return[0].shape
received        1 17 4            47            47 1 4
template        b  i r             k             p o r
   error                          ^^

padded_input_spatial (j) = [60] and block_shape (k) = [47].  padded_input_spatial must be divisible by block_shape

Dimensions computed as:
output_batch = product(block_shape) * batch
padded_input_spatial = padding_start + input_spatial + padding_end

p = product(k) * b
j = s + i + e

[47] = product([47]) * 1
[60] = [34] + [17] + [9]
```

Here, TensorFlow's error message is not bad.  However, there is no mechanism
for synching the documentation with the constraint that is violated.  On the
other hand, opschema's 'explain' clearly states:

```
Computed dimensions

output_batch = product(block_shape) * batch
padded_input_spatial = padding_start + input_spatial + padding_end
output_spatial = padded_input_spatial // block_shape

p = product(k) * b
j = s + i + e
o = j // k

Index predicates

padded_input_spatial must be divisible by block_shape
padded_input_spatial must be >= 0
output_spatial must be >= 0
output_batch must be >= 0
```

using names which are programmatically guaranteed to appear in the run-time
error messages.


# Schema Sections

`opschema` defines an op schema using a few basic concepts common to all ops.
To best illustrate these I'll illustrate them with the example of the
`tf.nn.convolution` schema.

    python -m opschema.cl explain tf.nn.convolution -i

```
Schema for tf.nn.convolution

Indexes

Index  Description           
b      batch                 
i      input spatial         
f      filter spatial        
g      dilated filter spatial
s      strides               
d      dilations             
k      filter input channel
j      output filter         
l      output channel        
o      output spatial        

Signatures

input  filters  strides  dilations  return[0]  data_format             
bki    fjl      s        d          blo        ['NCW', 'NCHW', 'NCDHW']
bik    fjl      s        d          bol        ['NWC', 'NHWC', 'NDHWC']

Index ranks

rank(b) in [1, 5]     
rank(i) in [1, 3]     
rank(f) = rank(i)     
rank(g) = rank(i)     
rank(s) = rank(i)     
rank(d) = rank(i)     
rank(k) = 1           
rank(j) = 1           
rank(l) = 1           
rank(o) = rank(i)     

Computed dimensions

dilated_filter_spatial = (filter_spatial - 1) * dilations + 1
output_spatial = ceil(input_spatial / strides)   [padding = SAME]
output_spatial = ceil((input_spatial + dilated_filter_spatial - 1) / strides)   [padding = VALID]

g = (f - 1) * d + 1
o = ceil((i + g - 1) / s)   [padding = VALID]
o = ceil(i / s)   [padding = SAME]

Index predicates

dilated_filter_spatial must be >= 0
output_spatial must be >= 0
strides and dilations dimensions cannot both contain an element over 1
input_channel must be divisible by output_filter

g must be >= 0
o must be >= 0
s and d dimensions cannot both contain an element over 1
k must be divisible by j

DType Rules

input.dtype in (int32, float16, float32, float64, bfloat16)
filters.dtype = input.dtype

Excluded DType Combos

input.dtype  rank(i)  layout
int32        1,2      0     
int32        3        *     
bfloat16     1,2      *     
bfloat16     3        0     

Inventory

input.shape  input.dtype  filters.shape  filters.dtype  strides  data_format  dilations  return[0].shape
bki          float16      fjl            float16        s        NCW          d          blo            
bki          float32      fjl            float32        s        NCW          d          blo            
bki          float64      fjl            float64        s        NCW          d          blo            
bik          int32        fjl            int32          s        NWC          d          bol            
bik          float16      fjl            float16        s        NWC          d          bol            
bik          float32      fjl            float32        s        NWC          d          bol            
bik          float64      fjl            float64        s        NWC          d          bol            
bki          float16      fjl            float16        s        NCW          d          blo            
bki          float32      fjl            float32        s        NCW          d          blo            
bki          float64      fjl            float64        s        NCW          d          blo            
bik          int32        fjl            int32          s        NWC          d          bol            
bik          float16      fjl            float16        s        NWC          d          bol            
bik          float32      fjl            float32        s        NWC          d          bol            
bik          float64      fjl            float64        s        NWC          d          bol            
bkii         float16      ffjl           float16        ss       NCHW         dd         bloo           
bkii         float32      ffjl           float32        ss       NCHW         dd         bloo           
...
```

`opschema` uses three abstractions to define the schema:  *index*, *signature*,
and *layout*.  The first section lists the indices:


## Index section

```bash
Index  Description           
b      batch                 
i      input spatial         
f      filter spatial        
g      dilated filter spatial
s      strides               
d      dilations             
k      input channel         
j      filter input channel 
l      output channel        
o      output spatial        
```
opschema Indexes are declared with
[add_index](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L899) 
as in:

```python
# excerpt from opschema/ops/tf/nn/convolution.py
# declare an index called 'batch' which can range in rank from 1 to 5
op.add_index('b', 'batch', (1,5))
op.add_index('i', 'input spatial', (1,3))

# declare index 'f' to have rank equivalent to index 'i'
op.add_index('f', 'filter spatial', 'i')
...
```

opschema `Index` objects represent shaped quantities.  They are not always
instantiated directly in input or output tensors, however.  Any quantities that
participate in computations that involve shapes, even intermediate
calculations, can be declared as `Index` entities.  In the example above,
'strides' and 'dilations' are ordinary parameters, while 'dilated filter
spatial' is an intermediate index that does not appear in any inputs or outputs
of the op.


## Signatures section

```bash
Signatures

input  filters  strides  dilations  return[0]  data_format             
bki    fjl      s        d          blo        ['NCW', 'NCHW', 'NCDHW']
bik    fjl      s        d          bol        ['NWC', 'NHWC', 'NDHWC']
```

This section shows a table with one *layout* for each row.  Each column
represents a shape-bearing parameter (which may be a tensor, but may not).  The cells in
the row define *signatures*, which are concatenations of the single letter
codes for `Index` objects.  For example, the 'filters' parameter has signature
'fjl', meaning that its shape is interpreted as a set of dimensions 'filter
spatial', then 'filter input channel', then 'output channel'.

The individual arguments are registered with the schema depending on the kind
of argument.  Input tensors are registered with [arg_tensor]( https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1499)
and return tensors with [return_tensor]( https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1776).
The signatures are declared with these API calls, and the layouts are
associated with the `data_format` parameter using the API call 
[arg_layout](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1389).

The OpSchema API calls are:

```python
# excerpt from opschema/ops/tf/nn/convolution.py
formats = {
        'NCW': (0, 1), # layout 0, rank(i) = 1
        'NCHW': (0, 2), # etc...
        'NCDHW': (0, 3),
        'NWC': (1, 1),
        'NHWC': (1, 2),
        'NDHWC': (1, 3),
        None: (1, None),  # default layout is layout 1, regardless of rank(i)
        }

# argument 'data_format' determines the layout according to the 'formats' map
# and the rank of index 'i'
op.arg_layout('data_format', formats, 'i')

# tensor 'input' is registered with signatures for each layout
op.arg_tensor('input', 'bki', 'bik')
op.arg_tensor('filters', 'fjl')
```

## Index ranks


```bash
Index ranks

rank(b) in [1, 5]     
rank(i) in [1, 3]     
rank(f) = rank(i)     
rank(g) = rank(i)     
rank(s) = rank(i)     
rank(d) = rank(i)     
rank(k) = 1           
rank(j) = 1           
rank(l) = 1           
rank(o) = rank(i)     
```

The Index ranks section defines rank constraints for each `Index` object.  An
Index rank means the same as for a tensor, but for a subset of semantically
related indices.  For instance, 'filter.rank' is equal to `rank(f) + rank(j) +
rank(l)`.  According to the above constraints, this would imply it could range
from 3 to 5.  All of the above rank constraints are determined during index
creation, but an additional API function [limit_ranks](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1246)
can be used.

## Computed dimensions


```bash
Computed dimensions

dilated_filter_spatial = (filter_spatial - 1) * dilations + 1
output_spatial = ceil(input_spatial / strides)   [padding = SAME]
output_spatial = ceil((input_spatial + dilated_filter_spatial - 1) / strides)   [padding = VALID]

g = (f - 1) * d + 1
o = ceil((i + g - 1) / s)   [padding = VALID]
o = ceil(i / s)   [padding = SAME]
```

The Computed dimensions section shows the formulas registered for Computed Indexes.
The formulas are shown in snake-cased
form and single-letter-code form.  For formulas that depend on other op
parameters (in this case the 'padding' parameter), the variants of the formulas
are shown.  These formulas are used both to compute valid inputs during error
checking, and to generate readable formulas for context in error messages.

Computed dimensions are registered with the API call [OpSchema.comp_dims](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1162)
and related variants.

```python
# excerpt from opschema/ops/tf/nn/convolution.py
from opschema.complib import dilate, dilate_t, strided_conv, strided_conv_t

# Index 'g' (dilated filter spatial) is computed using the dilate function
# from f (filter spatial) and d (dilation)
op.comp_dims_cw('g', dilate, dilate_t, 'fd') 

# Index 'o' (output spatial) is computed using the strided_conv function from 
# index 'i' (input spatial), 'g' (dilated filter spatial), and 's' (stride)
op.comp_dims_cw('o', strided_conv, strided_conv_t, 'igs', 'padding')
```

Because certain formulas recur in many ops, such functions may be found in
`opschema/complib.py`.  A numeric version operating on integers and a template
version interpolating string representations must be provided.  For example:

```python
# excerpt from opschema/complib.py
def strided_conv(i, f, s, padding):
    if padding == 'VALID':
        return ceildiv(i - f + 1, s)
    else:
        return ceildiv(i, s)

def strided_conv_t(i, f, s, padding):
    if padding == 'VALID':
        return f'ceil(({i} + {f} - 1) / {s})'
    else:
        return f'ceil({i} / {s})' 
```

Because the schema overall is defined as a python function, any custom compute
functions may be defined as local functions as well.  Placing them in
`opschema/complib.py` is just a convenience.

## Index Predicates

```bash
Index predicates

dilated_filter_spatial must be >= 0
output_spatial must be >= 0
strides and dilations dimensions cannot both contain an element over 1
input_channel must be divisible by filter_input_channel 

g must be >= 0
o must be >= 0
s and d dimensions cannot both contain an element over 1
k must be divisible by j
```

Predicate functions may be registered on individual or combinations of indices.
A non-negativity predicate is automatically registered on all computed indices.
In the above example, these are 'dilated filter spatial' and 'output spatial'.
The schema author may register additional predicates.  In the case of
`tf.nn.convolution`, 'input channel' must be disivible by 'filter input
channel'.  In fact this is not documented, but it is empirically true. 

Predicates are registered with API call
[dims_pred](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1710)
and its component-wise variant, as follows:

```python
# excerpt from opschema/ops/tf/nn/convolution.py
# only stride or dilation components can be over 1, not both (this is documented)
op.dims_pred('s-d exclusion', 
        predlib.not_both_over_one,
        predlib.not_both_over_one_templ, 'sd')

# input channel must be disivible by filter input channel (not documented)
op.dims_pred_cw('k % j == 0', predlib.divis_by, predlib.divis_by_t, 'kj')
```

## DType constraints

```bash
DType Rules

input.dtype in (int32, float16, float32, float64, bfloat16)
filters.dtype = input.dtype

Excluded DType Combos

input.dtype  rank(i)  layout
int32        1,2      0     
int32        3        *     
bfloat16     1,2      *     
bfloat16     3        0     
```

Constraints on allowed DTypes are given first as a set of broad rules, and then
specific exclusions.  The DType Rules can be one of two forms - either specify
that some tensor can take on certain dtypes, or specify that a tensor dtype
must be the same as another tensor.

The Excluded DType Combos section specifies combinations of dtype, index rank,
and possibly layout which are excluded.  Usually this is done because such
combinations are not implemented.  In the above example, `int32` Conv1D and
Conv2D are not implemented specifically for layout 0, which means data_formats
'NCW', 'NCHW'.

DType constraints are declared using API calls 
[valid_dtypes](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1269),
[equate_dtypes](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1305),
[exclude_combos](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1327)

as shown here:

```python
# excerpt from opschema/ops/tf/nn/convolution.py
op.valid_dtypes('input', ('int32', 'float', 'bfloat16'))
op.equate_dtypes('filters', 'input')
op.exclude_combos('input', 'int32', 'i', (1,2), LAYOUT, 0)
op.exclude_combos('input', 'int32', 'i', 3)
op.exclude_combos('input', 'bfloat16', 'i', (1,2))
op.exclude_combos('input', 'bfloat16', 'i', 3, LAYOUT, 0)
```

## Other Constraints

There are other relationships between inputs in certain TensorFlow ops.  For
example, with `tf.gather_nd`, the last dimension of the `indices` shape
determines the rank of the 'read location' (r) index.  This is declared using
the API function [rank_dims_constraint](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1698).
For a complete list of API functions, see `opschema.schema.OpSchema` class.

# Computation Graphs

The schema API internally builds four computation graphs.  They can be viewed
with:

    python -m opschema.cl graph OP_PATH OUT_DIR

This will produce pdf files `OUT_DIR/OP_PATH.{pred,gen,inf,dims}.pdf`.  A
computation graph here has the usual meaning - nodes wrap functions, and
the parents of a node provide the inputs to the function.  Nodes without
parents wrap functions that take no inputs.  Evaluating the graph as a whole
means evaluating the functions in valid topological order.

## Generative Graph

Two specializations of this idea are used in opschema.  A ***generative
graph*** has nodes which wrap generator functions, which are provided in
[opschema/generators.py](https://github.com/hrbigelow/opschema/blob/master/opschema/generators.py).
Each function will yield zero or more items, depending on the inputs it
receives.  The graph as a whole becomes a generator which yields value sets,
one value corresponding to each node.  This notion can be seen as a
generalization of `itertools.product`, which can be implemented as a generative
graph of fully disconnected nodes with no parents.

The `gen` graph is responsible for generating op input sets.  A subset of its
nodes represent parameters of the op, while another subset represent hidden
states which control relationships between them.

## Predicate Graph

The second specialization is a ***predicate graph***.  Its nodes wrap predicate
functions defined in
[opschema/predicates.py](https://github.com/hrbigelow/opschema/blob/master/opschema/predicates.py)
As before, nodes with no parents hold predicate functions (function objects
actually) which return a tuple `pred`, `data`.  If `pred` is True, `data` is
passed on to the node's children as an input argument and graph evaluation
proceeds.  If `pred` is False,
`data` is an instance of `ErrorReport` which holds information about the
failure, and graph evaluation halts.

