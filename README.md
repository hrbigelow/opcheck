
# opschema 

A system to build input constraint schemas for TensorFlow operations

Install from PyPI:

    pip install opschema
    pip install git+https://github.com/hrbigelow/opschema

# Motivation

TensorFlow Python is a workhorse of the Machine Learning world used by many
thousands of developers.  Tensor ops are often (necessarily) highly polymorphic with
intricate shape and other required relationships in inputs.  If these are not
met, often the exception will arise from several stack levels down the
codebase.  Because of this, it is frequently not clear to the user what input
constraints are violated and what should be done to correct the error.

User demand for more functionality is ever present, and so TensorFlow evolves
at a quick pace, presenting a challenge for developers to keep documentation
updated, as well as the many messages in exceptions to describe errors.

Documentation very often does not fully describe the legal inputs to ops. Finding
out whether a particular call is legal must be done by trial and error in many
cases.

# Introduction

opschema provides an API for building *op schemas* for representing TensorFlow
operations.  Once written, a schema represents a single operation, such as
`tf.nn.convoution` or `tf.nn.bias_add`, etc.  The schema defines what inputs are
legal for the op.  Once defined, it provides four functionalities:

* provide programmatically generated error messages using actual parameter
  names and identifiers registered in the schema as a single source of truth.

* provide programmatically generated documentation of legal call
  configurations, using actual parameter names, and the same identifiers used
  in error messages.

* generate a complete set of legal (and a particular set of illegal) inputs for
  the op, useful for comprehensive ops unit testing

* empirically validate schema correctness against TensorFlow op, given in TP,
  TN, FP and FN counts

# Proof of Concept repo only

opschema is intended only as a proof-of-concept, a demonstration of an idea
that might be integrated into TensorFlow.  In order to be viable outside of
TensorFlow, it would need to provide a set of schemas for every version of
TensorFlow, as ops were augmented and updated, which is unmaintainable.

Instead, the viable path would be to have an internal schema API integrated
into the the TensorFlow codebase, with each user-level op having a schema
definition which co-evolves as the op itself is augmented over the different
versions of TensorFlow.

# Synopsis

List available op schemas (defined under opschema/ops)

    python -m opschema.cl list

Programmatically generate documentation for an op

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

# Overview 

`opschema` provides an API for writing *schemas* for TensorFlow ops.  A schema
here means a set of rules that define what combinations of inputs are legal.
Once a schema is defined, you can use opschema to generate a complete set of
test inputs for the op for all legal combinations of tensor dtypes, shapes, and
combinations of other control arguments such as `data_format` etc.  In
addition, a subset of illegal inputs can be generated as well, which are useful
for comparing TensorFlow's exception with opschema's error message.

# Example Error Messages

This section gives many examples of calls to TensorFlow ops which raise an
exception.  Each example compares the TensorFlow exception text to the error
message provided by `opschema`.  All examples are generated using the command:

    python -m opschema.cl validate OP_PATH OUT_DIR --max_dtype_err=1

This command internally uses the schema graphs (described below) to generate
all possible combinations of valid (and many invalid) inputs for the op, and
then run a wrapped version of the op, collecting both `opschema` error message
and the TensorFlow exception text if any.

opschema provides error messages as a list of one or more 'fixes'.  Each fix
has an associated `FixKind`, which is a bitmask of the following enum:

```python
class FixKind(enum.Enum):
    DTypeEquate = 1    # input tensors have differing dtypes but should match
    DTypeIndiv = 2     # one input tensor has a disallowed dtype
    ComboExcluded = 4  # this combo of dtypes, index ranks and/or layout is excluded 
    InsertDim = 8      # fix by inserting dimension(s) to a particular tensor 
    DeleteDim = 16     # fix by deleting dimension(s) from a particular tensor
    IndexUsage = 32    # an index appearing multiple places has differing dimension
    IndexPred = 64     # an index predicate is violated
    Layout = 128       # fix by trying an alternate layout
```

Each example is given in the format:

```
## ID   CLASS    OP_PATH: ARGS_LIST
TensorFlow Exception
TENSORFLOW_EXCEPTION_TEXT

FixKind
OPSCHEMA_FIX

FixKind
OPSCHEMA_FIX
...
```

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

I have cut-and-pasted a selection of them from multiple ops together by
`FixKind` for illustration purposes.

In many of the messages, there will be tables using semantic, one-letter-codes
assigned to dimensions of tensor shapes.  These codes and tables are explained
in the [Schema](#schema) section below.  

## DTypeEquate examples

These errors occur when two input tensors' dtypes differ but should not.  In
this situation, TensorFlow assumes that the first dtype was the correct one and
suggests changing the second one.   opschema does not make this assumption.
Additionally, TensorFlow does not use actual parameter names in its error
messages, while opschema always does.

This constraint is declared using API function [equate_dtypes](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1318).

```
## 37   TP      tf.nn.conv_transpose: input=[61, 27, 91]:bfloat16, filters=[92, 2, 27]:float64, output_shape=[61, 270, 182], strides=1, padding=VALID, data_format=NCW, dilations=1
TensorFlow Exception
cannot compute Conv2DBackpropInput as input #2(zero-based) was expected to be a double tensor but is a bfloat16 tensor [Op:Conv2DBackpropInput]

DTypeEquate
Received filters.dtype = float64 and input.dtype = bfloat16.  dtypes of filters and input must match.

## 32   TP      tf.nn.convolution: input=[44, 52, 16]:float16, filters=[79, 26, 21]:qint16, strides=1, padding=VALID, data_format=NCW, dilations=1
cannot compute Conv2D as input #1(zero-based) was expected to be a half tensor but is a qint16 tensor [Op:Conv2D]

DTypeEquate
Received filters.dtype = qint16 and input.dtype = float16.  dtypes of filters and input must match.

## 13   TP      tf.nn.bias_add: value=[71, 8]:int32, bias=[8]:float32, data_format=NC..
TensorFlow Exception
cannot compute BiasAdd as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:BiasAdd]

DTypeEquate
Received bias.dtype = float32 and value.dtype = int32.  dtypes of bias and value must match.
```

## DTypeIndiv examples

DTypeIndiv fixes occur when a particular input tensor is declared with
[valid_dtypes](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1282), for example:

```python
# from opschema/ops/tf/nn/convolution.py
# input dtype may be int32, any kind of float, or bfloat16
op.valid_dtypes('input', ('int32', 'float', 'bfloat16'))
...
```

These errors tend to produce long TensorFlow exceptions that don't mention
which tensor had a disallowed dtype.  opschema mentions the tensor by name and
which dtypes are allowed.

```
## 55   TP      tf.nn.bias_add: value=[13, 77]:qint8, bias=[77]:qint8, data_format=NC..
TensorFlow Exception
Could not find device for node: {{node BiasAdd}} = BiasAdd[T=DT_QINT8, data_format="NCHW"]
All kernels registered for op BiasAdd:
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_INT32]
  device='CPU'; T in [DT_INT8]
  device='CPU'; T in [DT_UINT8]
  device='CPU'; T in [DT_INT16]
  device='CPU'; T in [DT_UINT16]
  device='CPU'; T in [DT_UINT32]
  device='CPU'; T in [DT_INT64]
  device='CPU'; T in [DT_UINT64]
  device='GPU'; T in [DT_INT32]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
 [Op:BiasAdd]

DTypeIndiv
Received value.dtype = qint8.  Valid dtypes for value are: int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16, complex64 and complex128

## 41   TP      tf.nn.avg_pool: input=[31, 9, 1]:int16, ksize=[1], strides=[43], padding=VALID, data_format=NCW
Value for attr 'T' of int16 is not in the list of allowed values: half, bfloat16, float, double
        ; NodeDef: {{node AvgPool}}; Op<name=AvgPool; signature=value:T -> output:T; attr=ksize:list(int),min=4; attr=strides:list(int),min=4; attr=padding:string,allowed=["SAME", "VALID"]; attr=data_for
mat:string,default="NHWC",allowed=["NHWC", "NCHW"]; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:AvgPool]

DTypeIndiv
Received input.dtype = int16.  Valid dtypes for input are: bfloat16, float16, float32 and float64

## 129  TP      tf.nn.space_to_depth: input=[17, 46, 2, 8]:qint16, block_size=45, data_format=NHWC
TensorFlow Exception
Could not find device for node: {{node SpaceToDepth}} = SpaceToDepth[T=DT_QINT16, block_size=45, data_format="NHWC"]
All kernels registered for op SpaceToDepth:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 16005131165644881776, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 16005131165644881776, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='GPU'; T in [DT_UINT8]
  device='GPU'; T in [DT_QINT8]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_QINT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_VARIANT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_RESOURCE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_STRING]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BOOL]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX128]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_DOUBLE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_FLOAT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BFLOAT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_HALF]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT64]; data_format in ["NHWC"]
 [Op:SpaceToDepth]

DTypeIndiv
Received input.dtype = qint16.  Valid dtypes for input are: bool, complex64, complex128, qint8, bfloat16, int8, int16, int32, int64, float16, float32, float64, uint8, uint16, uint32 and uint64
```

## ComboExcluded examples

ComboExcluded fixes are generated when a particular combination of input
dtypes, index rank and/or layout (as inferred by `data_format`) is given but
which was excluded by the schema API call [exclude_combos](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1340).  For example, 

```python
# from opschema/ops/tf/nn/convolution.py
# layout 0 here is designated by data_format NCW, NCHW or NCDHW 
op.exclude_combos('input', 'int32', 'i', (1,2), LAYOUT, 0)
op.exclude_combos('input', 'int32', 'i', 3)
op.exclude_combos('input', 'bfloat16', 'i', (1,2))
op.exclude_combos('input', 'bfloat16', 'i', 3, LAYOUT, 0)
```

In these situations, TensorFlow sometimes produces verbose error messages that
don't mention which parameters have a problem, or an unrelated error message.

```
## 1    TP      tf.nn.avg_pool: input=[66, 17, 1]:bfloat16, ksize=[7], strides=[50], padding=VALID, data_format=NCW
Tried to squeeze dim index 2 for tensor with 1 dimensions. [Op:Squeeze]

ComboExcluded
This combination is not implemented: input.dtype in (bfloat16) and [1] input_spatial dimensions

## 89   TP      tf.nn.depth_to_space: input=[78, 312, 1, 36]:qint8, block_size=6, data_format=NHWC
TensorFlow Exception
qint8 should be used with data_format NCHW_VECT_C. [Op:DepthToSpace]

ComboExcluded
This combination is not implemented: input.dtype in (qint8, qint16, qint32) and data_format in (NHWC, NCHW)

## 91   TP      tf.nn.depth_to_space: input=[94, 129, 3, 96]:qint32, block_size=4, data_format=NHWC
TensorFlow Exception
Could not find device for node: {{node DepthToSpace}} = DepthToSpace[T=DT_QINT32, block_size=4, data_format="NHWC"]
All kernels registered for op DepthToSpace:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 16005131165644881776, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 16005131165644881776, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='GPU'; T in [DT_QINT8]
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_VARIANT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_RESOURCE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_STRING]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BOOL]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX128]; data_format in ["NHWC"]
  device='CPU'; T in [DT_COMPLEX64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_DOUBLE]; data_format in ["NHWC"]
  device='CPU'; T in [DT_FLOAT]; data_format in ["NHWC"]
  device='CPU'; T in [DT_BFLOAT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_HALF]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT8]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT16]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT32]; data_format in ["NHWC"]
  device='CPU'; T in [DT_INT64]; data_format in ["NHWC"]
  device='CPU'; T in [DT_UINT64]; data_format in ["NHWC"]
 [Op:DepthToSpace]

ComboExcluded
This combination is not implemented: input.dtype in (qint8, qint16, qint32) and data_format in (NHWC, NCHW)
```

## InsertDim examples

InsertDim fixes are suggested fixes issued by opschema to insert (usually one)
dimensions to some input tensor.  They occur when the combination of input
tensor (or other shape-bearing inputs) ranks are not among the allowed rank
combinations.  These combinations are logically constructed from
lower-level constraints on ranks of `OpSchema.schema.Index` objects and
argument *signatures* (a notion from opschema), explained below.

The tricky thing about incompatible ranks is that there can be multiple fixes,
and it is not clear which is most likely the one the user will take.

In cases like this, TensorFlow makes a strong assumption about which argument
rank was in error, which can be misleading if it guesses wrong.  Some are quite
misleading or incorrect.  

```
## 7    TP      tf.nn.atrous_conv2d: value=[34, 71, 1, 15]:int32, filters=[79, 1, 11]:int32, rate=14, padding=VALID
TensorFlow Exception
dilations should be of length 1, 1 or 3. Received: dilations=[14, 14] of length 2

Received invalid configuration: value rank = 4, filters rank = 3 and rate rank = 1.  Closest valid configurations:

InsertDim
            value.shape   filters.shape   rate   return[0].shape
received   [34,71,1,15]       [79,1,11]   [14]
config 1        b i i k      => f f k l      r           b o o l

=> config 1: add 1 dimension to filters

## 25   TP      tf.nn.convolution: input=[11, 80, 17]:float16, filters=[10, 21]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
TensorFlow Exception
num_spatial_dims (input.shape.ndims - num_batch_dims - 1) must be one of 1, 2 or 3 but saw 0.  num_batch_dims: 2.

Received invalid configuration: input rank = 3, filters rank = 2 and data_format = NCW.  Closest valid configurations:

InsertDim
           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received    [11,80,17]         [10,21]         1           NCW           1
config 1         b k i        => f j l         s                         d             b l o

=> config 1: add 1 dimension to filters

## 27   TP      tf.nn.convolution: input=[62, 16]:float16, filters=[38, 19, 4]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
TensorFlow Exception
input must be 4-dimensional[62,1,16] [Op:Conv2D]

Received invalid configuration: input rank = 2, filters rank = 3 and data_format = NCW.  Closest valid configurations:

InsertDim
           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received       [62,16]       [38,19,4]         1           NCW           1
config 1      => b k i           f j l         s                         d             b l o

=> config 1: add 1 dimension to input

## 311  TP      tf.nn.bias_add: value=[33, 50]:uint16, bias=[67]:uint16, data_format=NC..
TensorFlow Exception
Must provide as many biases as the last dimension of the input tensor: [67] vs. [33,50] [Op:BiasAdd]

Received invalid configuration: value rank = 2, bias rank = 1 and data_format = NC...  Closest valid configurations:

InsertDim
           value.shape   bias.shape   data_format   return[0].shape
received       [33,50]         [67]          NC..
config 1      => b c s            c                           b c s

=> config 1: add 1 dimension to value
```

## DeleteDim examples

Like the `InsertDim` examples, `DeleteDim` fixes are issued when the submitted
combination of ranks is not one of the allowed combinations.  In these
situations, multiple suggested fixes may be issued.  This is unfortunately
rather verbose.  The principle followed here is to search the set of legal
combinations, and find those closest in 'edit distance' with the provided
inputs.  In such cases, alternate `data_format` may also be part of the
suggested fix.

In many of these situations, TensorFlow makes a strong assumption about what is
wrong, which may not be what the user ultimately wants to fix.

```
## 7    TP      tf.nn.avg_pool: input=[2, 13, 36, 4]:float16, ksize=[6], strides=[88], padding=VALID, data_format=NCW
TensorFlow Exception
Value for attr 'data_format' of "NCW" is not in the list of allowed values: "NHWC", "NCHW"
        ; NodeDef: {{node AvgPool}}; Op<name=AvgPool; signature=value:T -> output:T; attr=ksize:list(int),min=4; attr=strides:list(int),min=4; attr=padding:string,allowed=["SAME", "VALID"]; attr=data_for
mat:string,default="NHWC",allowed=["NHWC", "NCHW"]; attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE]> [Op:AvgPool]

Layout
           input.shape   ksize   strides   data_format   return[0].shape
received     2 13 36 4       6        88           NCW          2 13 1 0
template     b  c  i i       k         s          NCHW          b  c o o
   error                                          ^^^^

=> Change data_format to NCHW

Layout
           input.shape   ksize   strides   data_format   return[0].shape
received     2 13 36 4       6        88           NCW           2 1 1 4
template     b  i  i c       k         s          NHWC           b o o c
   error                                          ^^^^

=> Change data_format to NHWC

Received invalid configuration: input rank = 4 and data_format = NCW.  Closest valid configurations:

DeleteDim
           input.shape   ksize   strides   data_format   return[0].shape
received   [2,13,36,4]       6        88           NCW
config 1      => b c i       k         s                           b c o

=> config 1: remove 1 dimension from input

## 9    TP      tf.nn.avg_pool: input=[2, 36, 1]:float16, ksize=[5, 1], strides=[67], padding=VALID, data_format=NCW
TensorFlow Exception
ksize should be of length 1, 1 or 3. Received: ksize=[5, 1] of length 2

Received invalid configuration: input rank = 3, ksize rank = 2 and data_format = NCW.  Closest valid configurations:

DeleteDim
           input.shape   ksize   strides   data_format   return[0].shape
received      [2,36,1]   [5,1]        67           NCW
config 1         b c i    => k         s                           b c o

=> config 1: remove 1 dimension from ksize

## 27   TP      tf.nn.conv_transpose: input=[80, 24, 55]:bfloat16, filters=[1, 70, 7, 24]:bfloat16, output_shape=[80, 84, 124], strides=1, padding=VALID, data_format=NCW, dilations=1
TensorFlow Exception
Tried to squeeze dim index 2 for tensor with 1 dimensions. [Op:Squeeze]

Received invalid configuration: input rank = 3, filters rank = 4, output_shape rank = 3 and data_format = NCW.  Closest valid configurations:

DeleteDim
           input.shape   filters.shape   output_shape   strides   data_format   dilations   return[0].shape
received    [80,24,55]     [1,70,7,24]    [80,84,124]         1           NCW           1
config 1         b k i        => f j k          b l o         s                         d             b l q

=> config 1: remove 1 dimension from filters
```

## IndexUsage examples

These fixes are issued when certain dimensions between two different tensors or
other shapes are supposed to be equal but aren't.  TensorFlow exceptions often
do not provide actual names for arguments involved.  opschema automatically
generates a table showing the context of the mismatching dimensions and
actual parameter names.

```
## 4    TP      tf.nn.bias_add: value=[52, 63]:int32, bias=[4]:int32, data_format=NC..
TensorFlow Exception
Must provide as many biases as the channel dimension of the input tensor: [4] vs. 63 in [52,63] [Op:BiasAdd]

IndexUsage
           value.shape   bias.shape   data_format   return[0].shape
received         52 63            4          NC..              52 ?
template          b  c            c                             b c
   error            ^^            ^                               ^

=> channel (c) has inconsistent dimensions in value and bias. value.shape[1] = 63 and bias.shape[0] = 4
```

## IndexPred examples

These fixes are issued when an index predicate is violated.  An index predicate
is a constraint that the index dimensions (as opposed to rank constraints) must
satisfy.  See [Index section](#index-section)).  Such predicates are declared
using
[dims_pred](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1722)
and related API calls.  For example, `tf.nn.convolution` only accepts strides
over 1 if dilation is equal to 1, and vice versa.  And, in several other
convolution-related ops, the input channel must be evenly divisible by the
filter input channel.  These are declared in `tf.nn.convolution` as:

```python
# from opschema/ops/tf/nn/convolution.py
# non-component-wise predicate:  all of the components of s must be equal to 1
# if any one of d is above 1, and vice versa
op.dims_pred('s-d exclusion', 
        predlib.not_both_over_one,
        predlib.not_both_over_one_templ, 'sd')

# component-wise predicate - each component of k must be divisible by the 
# corresponding component of j
op.dims_pred_cw('k % j == 0', predlib.divis_by, predlib.divis_by_t, 'kj')
```

By default, computed dimensions (see [Computed
dimensions](#computed-dimensions) section) automatically have a non-negativity
predicate assigned to them, which can be seen in opschema error messages and
automatically generated documentation.

```
## 65   TP      tf.nn.convolution: input=[84, 21, 2]:float64, filters=[2, 4, 8]:float64, strides=1, padding=VALID, data_format=NCW, dilations=1
input depth must be evenly divisible by filter depth: 21 vs 4 [Op:Conv2D]

IndexPred
           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received       84 21 2           2 4 8         1           NCW           1            84 8 1
template        b  k i           f j l         s                         d             b l o
   error          ^^               ^

input_channel (k) = [21] and filter_input_channel (j) = [4].  input_channel must be divisible by filter_input_channel

## 68   TP      tf.nn.convolution: input=[28, 27, 15]:float64, filters=[18, 27, 29]:float64, strides=[1], padding=VALID, data_format=NCW, dilations=1
Computed output size would be negative: -2 [input_size: 15, effective_filter_size: 18, stride: 1] [Op:Conv2D]

IndexPred
           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received      28 27 15        18 27 29         1           NCW           1          28 29 -2
template       b  k  i         f  j  l         s                         d           b  l  o
   error                                                                                  ^^

output_spatial (o) = [-2].  output_spatial must be >= 0

Dimensions computed as:
dilated_filter_spatial = (filter_spatial - 1) * dilations + 1
output_spatial = ceil((input_spatial + dilated_filter_spatial - 1) / strides)   [padding = VALID]

g = (f - 1) * d + 1
o = ceil((i + g - 1) / s)   [padding = VALID]

[18] = ([18] - 1) * 1 + 1
[-2] = ceil(([15] + [18] - 1) / 1)   [padding = VALID]

## 2    TP      tf.nn.conv_transpose: input=[46, 17, 31]:bfloat16, filters=[39, 3, 17]:bfloat16, output_shape=[46, 196, 69], strides=1, padding=VALID, data_format=NCW, dilations=1
TensorFlow Exception
Tried to squeeze dim index 2 for tensor with 1 dimensions. [Op:Squeeze]

IndexPred
           input.shape   filters.shape   output_shape   strides   data_format   dilations   return[0].shape
received      46 17 31         39 3 17      46 196 69         1           NCW           1         46 196 69
template       b  k  i          f j  k       b   l  o         s                         d          b   l  q
   error                          ^            ^^^                                                   ^^^

output_channel (l) = [196] and filter_output_channel (j) = [3].  output_channel must be divisible by filter_output_channel

## 3    TP      tf.nn.conv_transpose: input=[46, 17, 2]:bfloat16, filters=[39, 28, 17]:bfloat16, output_shape=[46, 196, 69], strides=1, padding=VALID, data_format=NCW, dilations=1
TensorFlow Exception
Tried to squeeze dim index 2 for tensor with 1 dimensions. [Op:Squeeze]

IndexPred
           input.shape   filters.shape   output_shape   strides   data_format   dilations   return[0].shape
received       46 17 2        39 28 17      46 196 69         1           NCW           1         46 196 40
template        b  k i         f  j  k       b   l  o         s                         d          b   l  q
   error                                           ^^                                                    ^^

output_spatial_declared (o) = [69] and output_spatial_computed (q) = [40].  output_spatial_declared must equal output_spatial_computed

Dimensions computed as:
strided_input_spatial = (input_spatial - 1) * strides + 1
dilated_filter_spatial = (filter_spatial - 1) * dilations + 1
output_spatial_computed = strided_input_spatial + dilated_filter_spatial - 1   [padding = VALID]

n = (i - 1) * s + 1
g = (f - 1) * d + 1
q = n + g - 1   [padding = VALID]

[2] = ([2] - 1) * 1 + 1
[39] = ([39] - 1) * 1 + 1
[40] = [2] + [39] - 1   [padding = VALID]
```

## Layout examples

Layout fixes are issued by opschema when the closest valid configuration either
involves a different layout or a different value for the layout-associated
argument (usually `data_format`).  In many examples, TensorFlow properly
detects that rank combinations, but does not inspect the dimensions to
determine the most plausible intended layout.  opschema actively checks both
layouts against actual dimensions for compatibility and suggests any that are
within a given 'edit distance'.

In other cases, the rank implied by the provided data_format is correct.  Since
the data_format parameter controls the interpretation of indexes, using the
wrong data_format might lead to an apparent error in index predicates.  In
example 914 below, TensorFlow assumes that the input depth and filter depth
values are erroneous, but this could also be a case of the wrong data_format
provided.

opschema has a flexible method for searching most plausible fixes based on a
heuristic assignment of edit distance scores.  These could be adjusted, as well
as thresholds for which fixes to report.

```
## 155  TP      tf.nn.conv_transpose: input=[25, 28, 44]:float16, filters=[50, 2, 28]:float16, output_shape=[25, 80, 93], strides=1, padding=VALID, data_format=NCDHW, dilations=1
TensorFlow Exception
`data_format` must be 'NWC' or 'NCW'. Received: data_format=NCDHW

Layout
           input.shape   filters.shape   output_shape   strides   data_format   dilations   return[0].shape
received      25 28 44         50 2 28       25 80 93         1         NCDHW           1          25 80 93
template       b  k  i          f j  k        b  l  o         s           NCW           d           b  l  q
   error                                                                ^^^^^

=> Change data_format to NCW

## 914  TP      tf.nn.convolution: input=[99, 17, 1, 57]:int32, filters=[64, 1, 19, 8]:int32, strides=1, padding=SAME, data_format=NCHW, dilations=1
input depth must be evenly divisible by filter depth: 17 vs 19 [Op:Conv2D]

Layout
           input.shape   filters.shape   strides   data_format   dilations   return[0].shape
received    99 17 1 57       64 1 19 8         1          NCHW           1         99 17 1 8
template     b  i i  k        f f  j l       s s          NHWC         d d          b  o o l
   error                                                  ^^^^

=> Change data_format to NHWC
```

## False Negatives

False negative results occur when opschema does not detect a violation of
schema constraints, yet TensorFlow raises an exception.  Depending on your
interpretation, this could be considered a bug in the schema itself or
unintended behavior from TensorFlow.  Some examples are shown below.

```
## 4    FN      tf.nn.convolution: input=[87, 80, 3]:float16, filters=[1, 2, 3]:float16, strides=1, padding=VALID, data_format=NCW, dilations=1
No algorithm worked! [Op:Conv2D]

None

## 1729 FN      tf.nn.convolution: input=[82, 1, 2, 42]:int32, filters=[2, 21, 9]:int32, strides=1, padding=VALID, data_format=NWC, dilations=[1]
The Conv2D op currently does not support grouped convolutions for integer types. A grouped convolution was attempted to be run because the input depth of 42 does not match the filter input depth of 21 [Op:Conv2D]

None

## 1    FN      tf.nn.conv_transpose: input=[46, 17, 31]:bfloat16, filters=[39, 28, 17]:bfloat16, output_shape=[46, 196, 69], strides=1, padding=VALID, data_format=NCW, dilations=1
TensorFlow Exception
Tried to squeeze dim index 2 for tensor with 1 dimensions. [Op:Squeeze]

None

```

## Some notable earlier examples with interpretations (different format)

```
## 59   TP      tf.nn.convolution: input=[69, 45]:float32, filters=[59, 23, 2]:float32, strides=1, padding=VALID, data_format=NCW, dilations=1
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
## 98   FP      tf.nn.convolution: input=[83, 5, 90]:float32, filters=[90, 5, 25]:float32, strides=1, padding=VALID, data_format=NCHW, dilations=1
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

```
## 1    TP      tf.nn.avg_pool: input=[66, 17, 1]:bfloat16, ksize=[7], strides=[50], padding=VALID, data_format=NCW
Tried to squeeze dim index 2 for tensor with 1 dimensions. [Op:Squeeze]

This combination is not implemented: input.dtype in (bfloat16) and [1] input_spatial dimensions
```

Here, TensorFlow's exception does not give the user any clue what went wrong.
opschema's error message in my opinion could be improved, but is reasonably
clear.  Perhaps it could be augmented with a template table as in previous
examples.

```
## 9    TP      tf.nn.avg_pool: input=[2, 36, 1]:float16, ksize=[5, 1], strides=[67], padding=VALID, data_format=NCW
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
## 13   TP      tf.nn.avg_pool: input=[91, 19, 5]:float16, ksize=[5], strides=[14, 1], padding=VALID, data_format=NCW
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
## 42   FP      tf.nn.avg_pool: input=[37, 40, 6]:float16, ksize=[8], strides=[98], padding=VALID, data_format=NCHW
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


```
## 17   TP      tf.nn.depth_to_space: input=[90, 306, 1, 51]:int32, block_size=1, data_format=NHWC
Value for attr 'block_size' of 1 must be at least minimum 2
        ; NodeDef: {{node DepthToSpace}}; Op<name=DepthToSpace; signature=input:T -> output:T; attr=T:type; attr=block_size:int,min=2; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW", "NCHW_VECT_C"]> [Op:DepthToSpace]

Argument 'block_size' expected to be an integer >= 2
```

The first part of this message is not bad, except that it is confusing to call
it attr 'block_size'.  And, the remaining part is just visual noise.


```
## 7    TP      tf.nn.depth_to_space: input=[40, 36]:int8, block_shape=[31, 1], paddings=[11], [15]
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
## 8    TP      tf.nn.depth_to_space: input=[47, 34]:int8, block_shape=[], paddings=[1], [23]
paddings should have shape [0, 2] instead of [1,2] [Op:SpaceToBatchND]

Received invalid configuration: input rank = 2, block_shape rank = 0, paddings.0 rank = 1 and paddings.1 rank = 1.  Closest valid configurations:

           input.shape   block_shape   return[0].shape
received       [47,34]            []
config 1           b i          => k               p o

=> config 1: add 1 dimension to block_shape
```

Above, 'block_shape' was incorrectly provided with rank 0, when it must be
between 1 and 3.  But, TensorFlow incorrectly suggests to change the 'paddings'
parameter rank to match that of 'block_shape', leading to a nonsensical
suggestion.

```
## 21   TP      tf.nn.depth_to_space: input=[1, 17, 4]:int16, block_shape=[47], paddings=[34], [9]
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
other hand, opschema's 'explain' (detailed below) programmatically generates
documentation using the same names used in the generated error messages:

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

# Schema

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
[add_index](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L912) 
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
of argument.  Input tensors are registered with [arg_tensor]( https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1512)
and return tensors with [return_tensor]( https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1788).
The signatures are declared with these API calls, and the layouts are
associated with the `data_format` parameter using the API call 
[arg_layout](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1402).

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
creation, but an additional API function [limit_ranks](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1259)
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

Computed dimensions are registered with the API call [comp_dims](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1173)
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
[dims_pred](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1722)
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
[valid_dtypes](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1282),
[equate_dtypes](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1318),
[exclude_combos](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1340)

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
the API function [rank_dims_constraint](https://github.com/hrbigelow/opschema/blob/master/opschema/schema.py#L1711).
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

In addition, the `gen` graph is used during input evaluation.  It generates
potential interpretations of the inputs, and for each interpretation, measures
edit distance against the provided inputs.  In this way, opschema is capable of
detecting multiple potential fixes for a given set of erroneous inputs.  The
thresholds and edit distance scoring heuristics could be adjusted.

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

