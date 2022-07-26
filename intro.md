$
\newcommand\ein[1]{ \color{RoyalBlue}{\mathsf{#1}} }
\newcommand\aryid[1]{ \color{ForestGreen}{#1} }
\newcommand\sym[1]{ \color{black}{\mathtt{#1}} }
\newcommand\com[0]{ \sym{,\,} }
\newcommand\ary[2]{ \ein{ \aryid{#1} \sym{[} #2 \sym{]} } }
\newcommand\rankid[0]{ \color{firebrick}{RANK} }
\newcommand\dimsid[0]{ \color{firebrick}{DIMS} }
\newcommand\func[2]{ \ein{ \color{black}{#1} } \sym{(} #2 \sym{)} }
\newcommand{\flatid}[0]{FLAT}
\newcommand\rank[1]{ \ein{ \rankid \sym{(} #1 \sym{)} } }
\newcommand\dims[1]{ \ein{ \dimsid \sym{(} #1 \sym{)} } }
\newcommand{\flatcall}[1]{ \ein{\flatid \sym{(} #1 \sym{)} } }
$


# Introduction to the Einsum Tuple mini-language

I introduce a notation or mini-language which I call **Einsum Tuple** for describing tensor operations, available in the [einsum-tuple](https://github.com/hrbigelow/einsum-tuple) repo.  It is a high-level language in the style of MatLab, which is both close to an abstract mathematical form, and also executable by an interpreter.  The interpreter, or 'runtime' runs a short **Einsum Tuple** program, usually only three lines, defining inputs and producing outputs.  It then runs a user-specified TensorFlow operation on the same tensor inputs, and then verifies the **Einsum Tuple** outputs against those produced by the TensorFlow operation.

The **Einsum Tuple** program and the verifying TensorFlow operation call are specified together in a specially formatted `.et` (Einsum Tuple) file.  The file consists of four sections separated by a blank line:

1. **Einsum Tuple** program itself
2. The TensorFlow operation call, in mostly Python
3. The output tensor(s) to verify
4. Dimension and Rank constraints (explained later)

Each `.et` file serves the purpose of reverse-engineering one of the TensorFlow operations in the **Einsum Tuple** mini-language.  It serves to provide a verifiably correct, concise formula for that TensorFlow operation.  Several TensorFlow operations have been reverse-engineered in this way.  They can be found in the [ops directory](https://github.com/hrbigelow/einsum-tuple/tree/master/ops) of the repository.

Because many TensorFlow operations work with inputs of varying dimension, the **Einsum Tuple** runtime is built to instantiate the program with these different combinations of dimension numbers.  For example, the `tf.nn.convolution` operation can work over 1, 2, or 3 spatial dimensions, and the reverse engineered versions [ops/conv_valid_v1.et](https://github.com/hrbigelow/einsum-tuple/blob/master/ops/conv_valid_v1.et) and [ops/conv_valid_v2.et](https://github.com/hrbigelow/einsum-tuple/blob/master/ops/conv_valid_v2.et) automatically run all three cases.

## The formula for Convolution as a worked example

To introduce and motivate the **Einsum Tuple** language, I work through a familiar, but non-trivial example of the convolution operation, `tf.nn.convolution`, which is a general call that can work for 1, 2, or 3 spatial dimensions.

I start with the mathematical / pseudo-code description of the convolution computation given in the [documentation](https://www.tensorflow.org/api_docs/python/tf/nn/convolution).  This description, although mathematically complete, is not in a form which can actually be run and verified on a computer using real tensor inputs.  It is also somewhat verbose.  Step by step, I go through this notation and transform it, introducing three simplifications and certain notational rules along the way.  Each change still preserves the correctness of the formula.  Its final form is correct, and can be run and verified against the TensorFlow operation using the tool which implements this, called `eintup.py` from the [einsum-tuple](https://github.com/hrbigelow/einsum-tuple) repo.

We start with the formula given by TensorFlow documentation for [tf.nn.convolution](https://www.tensorflow.org/api_docs/python/tf/nn/convolution):

```
output[b, x[0], ..., x[N-1], k] =
    sum_{z[0], ..., z[N-1], q}
        filter[z[0], ..., z[N-1], q, k] *
        padded_input[b,
                     x[0]*strides[0] + dilation_rate[0]*z[0],
                     ...,
                     x[N-1]*strides[N-1] + dilation_rate[N-1]*z[N-1],
                     q]

N: number of spatial dimensions
b: batch index
x[i]: i'th output spatial index
z[i]: i'th filter spatial index
q: input channel index
k: output channel index
```                     

where the meanings of the indices are shown.  For the sake of obtaining a simplified example, I'll assume the special cases of this formula with stride = 1, dilation rate = 1, and a VALID convolution, in which the `padded_input` is identical to the original input.  Then, the formula simplifies to:

```
output[b, x[0], ..., x[N-1], k] =
    sum_{z[0], ..., z[N-1], q}
        filter[z[0], ..., z[N-1], q, k] *
        input[b, x[0] + z[0], ..., x[N-1] + z[N-1], q]
```

## Allow Indices to be Tuples

This formula is still somewhat difficult to read.  Being generic with respect to the number of spatial dimensions `N`, there are lots of nested brackets.  One way to simplify it and still retain the genericity of the formula, we could allow the identifiers `x` and `z` to represent tuples that obey component-wise arithmetic.  Let's assume this notation, so that:

```python
# Let x and z represent tuples with component-wise arithmetic
x     := x[0],        ..., x[N-1]
z     := z[0],        ..., z[N-1]
x + z := x[0] + z[0], ..., x[N-1] + z[N-1]
```

Using this more compact notation, the formula then simplifies to:

```python
# Using tuple-forms for x and z
output[b, x, k] = sum_{z, q} filter[z, q, k] * input[b, x + z, q]

# b: batch index
# x: output spatial index (a tuple of size N)
# z: filter spatial index (a tuple of size N)
# q: input channel index
# k: output channel index
```

where the meanings of the indices are given just below the formula.  

## Encourage use of meaningful index names

It looks like a nicer mathematical formula now, something in between math and pseudo-code.  But, as code, it looks awful!  One should not be encouraged to use so many single-letter identifiers that have no relation to their meaning.  The fact that it feels natural is just an arbitrary mathematical convention to use single-letter identifiers for indices.  Thinking more in terms of code, we could rewrite it as:

```python
# now with meaningful index names
output[batch, opos, ochan] = 
    sum_{fpos, ichan} filter[fpos, ichan, ochan]
                    * input[batch, opos + fpos, ichan]

# batch: batch index   (b)
# opos: output spatial index (a tuple of size N)  (x)
# fpos: filter spatial index (a tuple of size N)  (z)
# ichan: input channel index    (q)
# ochan: output channel index   (k)
```

where I've shown the former index names in parentheses.

Of course, the reader may decide whether the extra verbosity is worth it.  I favor the latter style, but this is a personal choice.  Using single-letter identifiers is still perfectly consistent within the notational rules.  

However, it's a mixed convention now, where the identifiers `batch`, `ichan`, and `ochan` represent integers, while `opos` and `fpos` represent tuples of length `N`.  From a language perspective this is bothersome because there is no way visually to tell the difference without using some font or other indicator.  But we could fix it by simply assuming that `batch`, `ichan`, and `ochan` are also tuples and later specify length 1.

Now, when the operation is actually dispatched in an ML framework the value of `N`, and the limits of summation are deduced from the known shapes and ranks of the tensor arguments (`input` and `filter`, in this case).  For example, `N` is calculated as `input.shape.rank - 2`.  The index ranges can be calculated as:

```python
batch < input.shape[0]
fpos[i] < filter.shape[i] # i in [0..N-1]
opos[i] < input.shape[i+1] # see note...
ichan < filter.shape[N]
ochan < filter.shape[N+1]
```

## Think in terms of Index Dimensions, not Tensor Dimensions

The above expressions are how an actual tensor operation would calculate index bounds, from the dimensions of tensor inputs.  However, most documentation describes dimensions in a more index-centric way.  Phrases like "a batch size of 10", or a "filter size of 9x9", or "3 input channels and 8 output channels", or "2 spatial dimensions" refer to index dimensions, not necessarily tensor dimensions.

For instance, the notion of "batch size" appears in both the input and output tensors.  We don't mention 'input' or 'output' when speaking of batch size, because the concept isn't specifically associated with one or the other.

Similarly, "number of input channels" is a property of both the input and filter.  Although it may be determined by the input, (hence the name), it defines part of the shape for both tensors.

Unfortunately, there is no notational mechanism to concisely express these phrases.  All statements concern either the size of a dimension (e.g. 'batch size of 10'), or a number of dimensions (e.g. 'two spatial dimensions').  

Conceptually, a solution to this is to endow *indices themselves* with the property of dimension sizes and numbers of dimensions.  And, then, think of the indices as determining the shapes of the tensors.  To this end, I propose two notational mechanisms called `DIMS(index)` and `RANK(index)`.  Using these constructs, one can more straightforwardly express the phrases above:

```
# Index-centric notation to express dimension size and number
DIMS(batch) = [10]     # "batch size of 10"
DIMS(fpos) = [9,9]     # "filter size of 9x9"
RANK(fpos) = 2         # "2 spatial dimensions"  (also, RANK(opos) = 2)
DIMS(ichan) = [3]      # "3 input channels"
DIMS(ochan) = [8]      # "8 output channels"
DIMS(opos) = [50,100]  # "output spatial size of 50x100"
```

Using these notations `DIMS()` and `RANK()` together with the compact formula, we can express the shape and rank of all tensors in a compact way:

```
output[batch, opos, ochan] = 
    sum_{fpos, ichan} filter[fpos, ichan, ochan]
                    * input[batch, opos + fpos, ichan]

# tensor shape is the concatenation (..) of its indices shape (DIMS)
output.shape = DIMS(batch) .. DIMS(opos) .. DIMS(ochan)
filter.shape = DIMS(fpos) .. DIMS(ichan) .. DIMS(ochan)
input.shape = DIMS(batch) .. DIMS(opos + fpos) .. DIMS(ichan)  # see note

# tensor rank is the sum of its indices' ranks
output.shape.rank = RANK(batch) + RANK(opos) + RANK(ochan)
filter.shape.rank = RANK(fpos) + RANK(ichan) + RANK(ochan)
input.shape.rank = RANK(batch) + RANK(opos + fpos) + RANK(ichan)
```

In fact, those last six formulas for tensor shape and rank are implied.  They follow automatically from the original expression.  Every tensor automatically takes on the shape of its indices in order.  And every tensor's rank is just the sum of the number of components (also known as the "rank") of the individual indices.

`DIMS(index)` means exactly the same as the `shape` field.  The i'th index of a tensor takes on values `[0, shape[i])`, that is, the half-open interval.  In this way, the shape field is the exclusive upper bound, or one more than the maximum value for that index.  The `DIMS()` construct is defined exactly the same:  `index in [0, DIMS(index))` with the semantics to be understood component-wise as `index[i] in [0, DIMS(index)[i]) for i in [0,RANK(index))`

With this definition, the expression `DIMS(opos + fpos)` is straightforward.  The component-wise *maximal values* of the expression `opos + fpos` will actually be `[49+8,99+8] = [57,107]`, therefore `DIMS(opos + fpos) = [58,108]`.  This is exactly what the `input` spatial shape must be for a VALID convolution to produce the output shape of `DIMS(opos) = [50,100]` using a 9x9 filter. 

## Implicit Summation (inspired by Einstein Summation notation)

As a final simplification, I take the idea of *implicit summation* from Einstein Summation notation.  In this idea, in an assignment statement, indices that appear on the right-hand-side but not on the left-hand-side are implicitly summed out and eliminated.  A good way to justify this is to think that the indices *disappear* when going from the right to the left.  And, the way to make a dimension of a tensor disappear is to sum over it.

Looking at the formula again:

```
output[batch, opos, ochan] = 
    sum_{fpos, ichan} filter[fpos, ichan, ochan]
                    * input[batch, opos + fpos, ichan]
```

note that the sum is over `fpos, ichan`, which just happen to be precisely the two indices that appear on the right but not on the left.  Every other index appears on both sides.  So, adopting the convention, we arrive at the final form of this notation:

```
# implicit summation, semantically named indices, and tuple indices
output[batch, opos, ochan] = filter[fpos, ichan, ochan]
                           * input[batch, opos + fpos, ichan]
```

## Summary of Einsum Tuple mini-language

In summary, we started with a general tensor assignment statement given in TensorFlow documentation, an expression given to mathematically define the operation.  Then we perform four transformations.

1. Group related sets of indices into tuple indices
2. Use meaningful index names, not single letters
3. Introduce the DIMS() and RANK() notation to specify index shapes
4. Make summation implicit

In the worked example, here is the before and after:

```python
# TensorFlow documentation formula for Convolution (N-D)
output[b, x[0], ..., x[N-1], k] =
    sum_{z[0], ..., z[N-1], q}
        filter[z[0], ..., z[N-1], q, k] *
        input[b, x[0] + z[0], ..., x[N-1] + z[N-1], q]

# Einsum Tuple formula for Convolution
output[batch, opos, ochan] = filter[fpos, ichan, ochan]
                           * input[batch, opos + fpos, ichan]
```

Further, in order to introduce restrictions for numbers or sizes of dimensions into the documentation, we can use the `DIMS()` and `RANK()` constructs.  In this case, convolution is only supported for 1, 2, or 3 spatial dimensions.  This can be expressed as `RANK(opos) IN [1, 3]`.  





# Case Study #2: `tf.gather_nd`

The TensorFlow operation [tf.gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd) is a tricky one to understand.  Partly this is because it is highly polymorphic in the combinations of batch dimensions and slicing that it allows.  The documentation has 15 different examples of its usage.

The underlying concept is very simple however.  An `indices` tensor is used as a mapping from *write addresses* to *read addresses*.  The element at *read address* in `params` is then written to `result` at *write address*.

There are three tricky parts, however.  First, the "element" that is read and then written may be a simple scalar, or it may be multi-dimensional, depending on the dimensions of the arguments.  Second, the "addresses" themselves, being indices, are multi-dimensional since they index multi-dimensional tensors.  Third, there is an optional batching logic.  Because of this, you must explicitly provide the number of batch dimensions since it is ambiguous in most cases.

To describe this operation in complete detail, I first introduce the idea of a tensor as a container of multi-dimensional objects rather than just scalars.  That interpretation will be needed to understand both the `indices` and `params` arrays.

Having established that, we then look at the structures of the `indices`, `params` and `result` tensor and how they fit together.  The notion of conceptual grouping of dimensions is central to this description.

## Tensors as containers of objects

A tensor is a multidimensional container of scalars.  But more generally, it can be interpreted as a container of *objects* which themselves can be tensors or scalars.  There are exponentially many interpretations of this kind.  But, if we restrict it so that the outer dimensions are interpreted as the container dimensions, and the inner dimensions as the dimensions of the *contained object*, then there are `N+1` interpretations.  For example:

```python
ten = tf.random.uniform(shape=[2,5,7,4,3], minval=0, maxval=10, dtype=tf.int32)
print(ten[:,:,:,:,:].shape)   # (2,5,7,4,3)  # 0D "container" of 5D tensors
print(ten[0,:,:,:,:].shape)   #   (5,7,4,3)  # 1D container of 4D tensors
print(ten[0,0,:,:,:].shape)   #     (7,4,3)  # 2D container of 3D tensors
print(ten[0,0,0,:,:].shape)   #       (4,3)  # 3D container of 2D tensors
print(ten[0,0,0,0,:].shape)   #        (3,)  # 4D container of 1D tensors
print(ten[0,0,0,0,0].shape)   #          ()  # 5D container of 0D tensors (scalars)
```

To facilitate particular interpretations, it would be useful to be able to index the tensor using named groups of indices.  I introduce a wrapper class that allows this indexing.

```python
# Wrap a tensor so that it can be indexed with tuples or scalars
# For example:  ten[(1,2),(0,1),:]
class TupTensor(object):
    def __init__(self, ten):
        self.ten = ten

    def _flatten(self, idx):
        for ent in idx:
            try:
                yield from ent
            except TypeError:
                yield ent

    def __getitem__(self, idx):
        inds = tuple(self._flatten(idx))
        return self.ten[inds]

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.ten, attr)

batch = (1,3)
source = (2,1)
ten = TupTensor(ten)

print(ten[:,:,:,:,:].shape)                  # (2,5,7,4,3)
# use named, tuple indices to conceptually group them
print(ten[batch,source,:].shape)             #         (3,)   (alternate indexing)

# Interpreted as a 4D tensor of 1D objects
print(ten[batch,source,:].numpy().tolist())  # [5, 3, 7] 
```

The alternate form of indexing is just a syntactic sugar to make it easier to mentally group dimensions into conceptual groups.  Because one of the indices is left as a wildcard slice ':', the returned "element" is a 1-D tensor with 3 elements.  This grouping of dimensions is essential for understanding the operation of `tf.gather_nd`.

## The tensor dimension structure of the `tf.gather_nd` operation.

Using this tuple indexing convention, we can write out the `tf.gather_nd` operation succinctly using pseudocode.

```
indices[batch, write_addr, :] = read_addr  # (a k-element read address)
params[batch, read_addr, :] = a d-dimensional "element" with shape elem_shape

# retrieve the element from params, write it to result
result[batch, write_addr, :] = params[batch, indices[batch, write_addr, :], :]
```

Here, both `indices` and `params` are indexed using the wildcard ':' at the end.  For `indices`, the wildcard takes up exactly one dimension.  Here, the 1D object represents the read_addr with k components.  For `params`, the ':' wildcard is not an address but rather the index into the scalar members of the d-dimensional object to be copied. 

In the final statement, the `results` tensor is also indexed with ':' at the end, which corresponds in shape with that of `params`.  This indicates that the d-dimensional object is copied scalar-for-scalar, preserving its shape.

From the tuple indices, we can see that `indices` has `len(batch) + len(write_addr) + 1` dimensions.

Note that `params` and `result` have almost identical structure.  Both start with batch dimensions and end with `d` element dimensions.  However, in the middle, `params` has `read_addr`, a set of k-dimensions.  `results` has `write_addr` which need not be the same number of dimensions.  The shape of the `write_addr` dimensions is determined from the `indices` tensor.

In my terminology, an 'element' is the d-dimensional unit of content that is moved together as one shape.  If d is zero, then it is just a scalar copy.  So, in the zero-dimensional case, `result` has the shape of `batch, write_addr`, and `params` has shape `batch, read_addr`, and the last line above is a scalar-to-scalar copy.  In the case where d > 0, it is conceptually a sub-tensor to sub-tensor copy.

The polymorphism of `tf.gather_nd` comes from the fact that there are many allowed combinations of numbers of dimensions (ranks) of `batch`, `read_addr`, `write_addr`, and `elem_shape`.  `len(batch)` and `len(elem_shape)` can be zero or more, while `len(read_addr)` and `len(write_addr)` must be at least 1.  The total number `len(batch) + len(read_addr) + len(elem)` must be less than or equal to 8, as it turns out.  All in all this is over 100 possible combinations of dimension structure.

## Total number of available read and write addresses

One point to build intuition about this operation is to consider the relation between the total number of read addresses and write addresses, equal to the product of dimensions for `read_addr` and `write_addr`.  If there are more read addresses than write addresses, then not all elements from `params` will be copied.  If there are more write addresses, then it is inevitable that some elements in `params` will be copied to multiple locations in `results`.

## Out of bounds read_addr

It is possible for the slices in the `indices` tensor to be out of bounds with respect to the `params` dimensions `read_addr`.  When this happens, it is silently ignored and no copy happens.  Since the `result` tensor is first initialized to zeros, this means it will remain zero for that entry in the `indices` array.



## Index Expressions are central to tensor operations

As a simple example of an index expression, consider the following routine, which takes the 5-element array `source` and builds a matrix from it, spreading the elements out diagonally:

```python
source = [1,4,7,5,2]

def index_expr(row, col):
  return col-row+2

# returns the ary element, or 0 if out of bounds
def safe_get(idx, ary):
  if idx in np.ndindex(ary.shape):
    return ary[idx]
  else:
    return 0

target = np.fromfunction(np.frompyfunc(safe_get(

# an 8x8 matrix of zeros
nrow = ncol = 8
target = [ [0] * ncol for _ in range(nrow) ]

for row in range(nrow):
  for col in range(ncol):
    index_expr = col-row+2
    if index_expr in range(len(source)):
      target[row][col] = source[index_expr]

for trg_row in target:
  print(repr(trg_row))
[7, 5, 2, 0, 0, 0, 0, 0]
[4, 7, 5, 2, 0, 0, 0, 0]
[1, 4, 7, 5, 2, 0, 0, 0]
[0, 1, 4, 7, 5, 2, 0, 0]
[0, 0, 1, 4, 7, 5, 2, 0]
[0, 0, 0, 1, 4, 7, 5, 2]
[0, 0, 0, 0, 1, 4, 7, 5]
[0, 0, 0, 0, 0, 1, 4, 7]
```

We can see that the elements of `source` have been picked up and placed in multiple locations in `target`, according to the index expression `row-col+2`.  To better visualize what this index expression looks like, we can also explicitly calculate it:

```python
indices = np.fromfunction(lambda row, col: col-row+2, [8,8], dtype=np.int32)
print(indices)
[[ 2  3  4  5  6  7  8  9]
 [ 1  2  3  4  5  6  7  8]
 [ 0  1  2  3  4  5  6  7]
 [-1  0  1  2  3  4  5  6]
 [-2 -1  0  1  2  3  4  5]
 [-3 -2 -1  0  1  2  3  4]
 [-4 -3 -2 -1  0  1  2  3]
 [-5 -4 -3 -2 -1  0  1  2]]

# Now we can instead use indices as the index expression
for row in range(nrow):
  for col in range(ncol):
    if indices[row][col] in range(len(source)):
      target[row][col] = source[indices[row][col]]
      # target[row][col] = source[col-row+2]
```

We have now introduced a layer of indirection.  Instead of indexing `source` using the expression `col-row+2` directly, we first stored that expression in an `indices` array, and later retrieved it.  

Here, `indices[row][col]` is being used as a function, like `indices(row, col)` its elements carry no information other than the expression value.





# Einsum Tuple - a mini-language for defining tensor operations

## Introduction

Einsum Tuple is a mini-language for defining tensor operations.  It resembles Einstein summation notation but has some additional properties.  It aims to provide a way to define fundamental tensor operations in a rank-agnostic way.  That is, one expression in Einsum Tuple may define a family of tensor operations, all sharing the same basic logic, but varying in the numbers of batch, spatial, channel, etc, dimensions.  This is a companion article to the source code at the [einsum-tuple](https://github.com/hrbigelow/einsum-tuple) repository.  To get a quick look at examples, see the [ops directory](https://github.com/hrbigelow/einsum-tuple/tree/master/ops).

Einsum Tuple language is an extension of Einstein Summation (einsum) notation, with these rules.

1. indices are tuples of unspecified length
2. tensors can be indexed with arbitrary expressions of indices (*index expressions*)
3. out-of-bounds index values are silently ignored
4. like einsum notation, broadcasting and summation are automatic
5. unlike einsum notation, indexing expressions appear in brackets, not subscripts

For example, here is a rank-agnostic definition of batched matrix multiplication:

$
\ary{output}{batch \com row \com col} = 
\ary{input}{batch \com row \com inner} \sym{*}
\ary{weights}{batch \com inner \com col}
$

The identifiers $\ein{batch}$, $\ein{row}$, $\ein{col}$, and $\ein{inner}$ are "eintups", the Einsum Tuple equivalent of Einstein indices.  Eintups symbolize a tuple of an unspecified number (even zero) of individual indices.  For instance:

\begin{array}{ll}
\ein{\aryid{output}}_{brc} & = \ein{\aryid{input}}_{bri} \ein{\aryid{weights}}_{bic} \\ 
\ein{\aryid{output}}_{b_{1}b_{2}rc} & = \ein{\aryid{input}}_{b_{1}b_{2}ri} \ein{\aryid{weights}}_{b_{1}b_{2}ic} \\ 
\ein{\aryid{output}}_{b_{1}b_{2}b_{3}rc} & = \ein{\aryid{input}}_{b_{1}b_{2}b_{3}ri} \ein{\aryid{weights}}_{b_{1}b_{2}b_{3}ic} \\ 
\ein{\aryid{output}}_{br_{1}r_{2}c} & = \ein{\aryid{input}}_{br_{1}r_{2}i} \ein{\aryid{weights}}_{bic} \\ 
...
\end{array}

In fact, it is more general than just matrix multiplication.  In ordinary batched matrix multiplication, each batch entry is strictly a rank-2 object, because the *row*, *inner* and *column* dimensions are single dimensions.  In the statement above however, there is no restriction on the ranks of $\ein{row}$, $\ein{col}$ or $\ein{inner}$, and yet, the statement is well defined and produces a result.


## Rank and Dimensions 

Before execution, the runtime system configures shapes of all eintups.  It does this in two phases, first choosing a *rank*, or number of indices, and then generates *dimensions*.  For example, the system might set the rank of batch to 3 and its dimensions to `[2,4,3]` before execution.  These quantities can be accessed in the language as $\dims{batch}$ and $\rank{batch}$.  They are constants during the execution phase, but change at each new shape configuration.

After shape configuration, the shapes of every array expression is known.  For example, the shape of $\ein{\aryid{output}}$  would be $\dims{batch \com row \com col}$, which is shorthand for $\dims{batch}$ concatenated with $\dims{row}$ and then $\dims{col}$.  Its rank is given by $\rank{batch \com row \com col}$ .

## Index expressions

Index expressions are arithmetic expressions of eintups and integer valued arguments, and sometimes functions of them.  For example, here is the `tf.nn.convolution` operation expressed in Einsum Tuple, using the `padding=VALID` option.

\begin{aligned}
\ary{output}{batch \com opos \com ochan} \sym{=\,} & 
\ary{filters}{ipos \sym{-} \dims{stride} \sym{*} opos \com ichan \com ochan} \\
\sym{*\,} & \ary{input}{batch \com ipos \com ichan}
\end{aligned}


The index expression is $\ein{ipos} \sym{-} \dims{stride} \sym{*} \ein{opos}$.  To be well-formed, binary operation between eintups and $\dims{\cdots}$ arguments must have equal rank.  In this case, $\ein{ipos}$, $\ein{stride}$, and $\ein{opos}$ must have the same rank or the statement won't compile.

Note that this expression can have component values that are negative.  Einsum Tuple implicitly ignores out-of-bounds indices (negative or too high).  If any component of an index expression in the top level assignment statement is out of bounds, the whole calculation doesn't participate in the Einstein summation / assignment.

In the statement above, it is clear that the convolution operation is linear in the $\ein{\aryid{input}}$ tensor since it appears with a single multiplicative term.  An equivalent formula showing linearity in the $\ein{\aryid{filters}}$ is given by:


\begin{aligned}
\ary{output}{batch \com opos \com ochan} & \sym{=}
\ary{filters}{fpos \com ichan \com ochan} \\
& \sym{*} \ary{input}{batch \com fpos \sym{+} \dims{stride} \sym{*} opos \com ichan}
\end{aligned}


## Indexing expression Basis

The indexing expression $\ein{ipos} - \dims{stride} \sym{*} \ein{opos}$ can be thought of as a computed tensor over the space $\dims{ipos \com opos}$  whose elements are 1D tuples with $\rank{ipos}$ members.  Each element of this virtual tensor is then used to index into its parent array $\ein{\aryid{filters}}$ , which expects a tuple of that size.

The set $\ein{ipos \com opos}$ of eintups is known as the *basis* for the indexing expression, and it is derived as the set of all eintup variables in the expression.  Note that while $\ein{stride}$ appears, it isn't a variable because $\dims{stride}$ resolves to a constant at runtime.

The basis of the full index list in the expression $\ary{filters}{ipos \sym{-} \dims{stride} \sym{*} opos \com ichan \com ochan}$ is then $\ein{ipos \com opos \com ichan \com ochan}$.  This is one eintup larger than the basis of the $\ein{\aryid{filters}}$ array to begin with.  Thus, one can think of this as a sort of calculated broadcast, or 'unrolling' of the filters tensor but in a diagonal direction.  The convolution becomes a fully connected layer "matrix multiplication" using this unrolled filter matrix.

In the second form, the input is 'unrolled' and becomes a matrix which multiplies the filters.

## Implicit Broadcasting

Einsum Tuple statements perform implicit broadcasting of tuples which appear on the left hand side of an assignment but not on the right.  For example, here is a formula for `tf.meshgrid`, which constructs a set of broadcasted tensors in a coordinated way.

\begin{aligned}
\ary{in1}{a} & = \func{RANDOM}{0, 10, \mathsf{INT}} \\
\ary{in2}{b} & = \func{RANDOM}{0, 10, \mathsf{INT}} \\
\ary{in3}{c} & = \func{RANDOM}{0, 10, \mathsf{INT}} \\
\ary{in4}{d} & = \func{RANDOM}{0, 10, \mathsf{INT}} \\
\ary{out1}{a \com b \com c \com d} & = \ary{in1}{a} \\
\ary{out2}{a \com b \com c \com d} & = \ary{in2}{b} \\
\ary{out3}{a \com b \com c \com d} & = \ary{in3}{c} \\
\ary{out4}{a \com b \com c \com d} & = \ary{in4}{d} \\
\end{aligned}

The $\ein{\aryid{out}}$ arrays are equivalent to the call `tf.meshgrid(in1, in2, in3, in4, indexing=L('ij'))`

In the assignment, $\ein{\aryid{out1}}$ receives broadcasted values for eintups $\ein{b}$, $\ein{c}$, and $\ein{d}$, and so forth.



## The FLAT() function

Aside from arithmetic binary operations, there is one function (so far) which accepts an index expression list and returns an index expression.  It returns a tensor of the same basis as its expression list.  Each element of the expression list is mapped into the flattened space of $\dims{expr\_list}$.  For example, if the index expression list has $\dims{expr\_list} = [2,4,3,7,2]$, then each element $(i,j,k,l,m)$ is mapped to the scalar quantity $i*4*3*7*2 + j*3*7*2 + k*7*2 + l*2 + m$.  This can be thought of as the index in the flat representation, assuming outer-dimension-major ordering of the tensor elements.

Flattening a multi-dimensional tensor with $\ary{output}{\func{FLAT}{a \com b \com c \com d}} = \ary{input}{a \com b \com c \com d}$ is equivalent to `output = tf.reshape(input, -1)`.  

Using this function, here is the Einsum Tuple expression for `tf.nn.space_to_depth`:

$
\ary{output}{batch \com ipos \sym{//} \dims{bsz} \com 
\func{FLAT}{ipos \sym{\,\%\,} \dims{bsz} \com ichan}} =
\ary{input}{batch \com ipos \com ichan}
$

This is a concise, complete description of the space-to-depth operation.  It is general with respect to $\rank{ipos}$, the number of spatial dimensions.  Also, the blocksize, $\dims{bsz}$ can be any shape, not necessarily square.  Tensorflow's implementation assumes two spatial dimensions and square blocksize.

It is instructive to reconcile the Einsum Tuple expression with the TensorFlow [documentation](https://www.tensorflow.org/api_docs/python/tf/nn/space_to_depth).  Here are some excerpts from `tf.nn.space_to_depth`:

> Non-overlapping blocks of size block_size x block size are rearranged into depth at each location.

The expression $\ein{ipos} \sym{//} \dims{bsz}$ takes on distinct values in the pattern of 'non-overlapping blocks` of the input locations.

> The Y, X coordinates within each block of the input become the high order component of the output channel index.

The expression $\ein{ipos \: \sym{\%\,} \dims{bsz}}$ is the 'Y, X coordinates within each block', and it appears as the high order component in the overall expression $\func{FLAT}{\ein{ipos} \: \sym{\%\,} \dims{bsz} \com \ein{ichan}}$.

> The depth of the output tensor is block_size * block_size * input_depth.

The expression $\ein{ipos \: \sym{\%\,} \dims{bsz}}$ takes on values up to the exclusive range $\dims{bsz}$.  `input_depth` corresponds to $\ein{ichan}$ and finally, the $\func{FLAT}{}$ call creates values in the range of the product of dimensions.

# Using an Array as Index Expression

An Eintup array of $n$ dimensions with integer elements may be viewed as a an array of $n-1$ dimensions whose elements are 1D integer tuples of a fixed size.  For example, let's assume an integer valued array $\ary{indices}{slice \com coord}$ of shapes $\dims{slice} = [3,5]$, $\dims{coord} = [7]$ so that the full shape of $\ein{\aryid{indices}}$ is $[3,5,7]$.  We can view the array itself as a function of two arguments (the component values of $\dims{slice}$) which outputs 7-tuples of integers.  The space over which the two arguments vary, are the dimensions $[3,5]$.  Using it as an index expression, we have:

\begin{aligned}
\ary{indices}{slice \com coord} & = \func{RANDOM}{0, 10, \mathsf{INT}} \\
\ary{output}{\ary{indices}{slice \com \sym{:}} \com elem} & = \cdots 
\end{aligned}

The index expression is $\ary{indices}{slice \com \sym{:}}$.  It is like an ordinary array access, except that $\ein{coord}$ has been called with the special "$\sym{:}$" symbol.  In order to be legal, $\rank{coord}$ must equal 1, and $\dims{coord}$ must equal the rank of the first place in the $\ein{\aryid{output}}$.  Note that it would be perfectly valid if the "$\sym{:}$" were in a non-terminal position.  For example, using $\ary{indices}{coord \com slice}$ as the array, and $\ary{indices}{\sym{:} \com slice}$ as the index expression is also valid.

Using an array slice as an index expression is a `scatter` operation if used on the left hand side, and a `gather` operation if on the right.

# Runtime

The `.et` file third section is the 'TF Call'.  It is a quasi-Python function call statement which is preprocessed in a few ways:

1. Bare identifiers resolve to tensors mentioned in the Einsum Tuple program
2. $\dims{\cdots}$ expressions resolve to a Python list of integers
3. $\rank{\cdots}$ expressions resolve to a Python integer
4. The $\func{L}{\cdots}$ function allows passing in a Python literal
5. The $\func{TENSOR}{\cdots}$ function creates a tensor from $\dims{}$, $\rank{}$ or integer arguments.

The runtime will run the TF Call, and then compare the output(s) with those listed on the Outputs line, which are computed using the Einsum Tuple program.

# Constraints

Constraints on the ranks and dimensions of all Eintups are specified in the fourth and last section of the `.et` file.  These constraints are used by the runtime to generate valid combinations of ranks and dimensions to instantiate the program and run it.

There are four types:


\begin{array}{ll}
\dims{tup} & \mathsf{IN} \: [\mathsf{min}, \mathsf{max} ] \\
\rank{tup} & \mathsf{IN} \: [\mathsf{min}, \mathsf{max} ] \\
\dims{tup} & \sym{=} \: \mathit{dcons\_expr} \\
\rank{tup} & \sym{=} \: \mathit{rcons\_expr} \\
\end{array}


The 'IN' forms cause the runtime to generate some value in `[min, max]` randomly.  The '=' forms evaluate the *dcons_expr* or *rcons_expr* and assign the result to the left hand side.  *rcons_expr* is an arithmetic expression that can consist of integers or $\rank{tup}$ expressions.  Taken together, these constraints form a dependency graph, and it is an error if the graph is circular.  The runtime system automatically finds the proper order to resolve the dependencies.

*dcons_expr* is an arithmetic expression of integers, $\dims{tup}$ or $\rank{tup}$ expressions.  Like the *rcons_expr*, it is an error if there are circular dependencies.  Note that the $\dims{}$ constraints can depend on both $\dims{}$ and $\rank{}$ expressions, while the $\rank{}$ constraints may only depend on other $\rank{}$ constraints.  This is because the runtime constraint resolution process proceeds in two phases.  In the first phase, it assigns all EinTup ranks according to the rank constraints, without assigning dimensions to the EinTups.  In the second phase, it resolves all dims constraints.

# Einsum Tuple Full Grammar Specification

The full grammar specification can be found in [eintup_grammar.ebnf](https://github.com/hrbigelow/einsum-tuple/blob/master/eintup_grammar.ebnf)
