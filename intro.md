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



# Index Expressions and Unrolling

The file [index_expr.py](https://github.com/hrbigelow/einsum-tuple/blob/master/doc/index_expr.py) implements two concepts useful for understanding the expressions of **Einsum Tuple** language.  The first is the notion of *unrollin* an index or index expression.  The second is the notion of *mapping* the unrolled index values to an array (or tensor).

As mentioned previously, **Einsum Tuple** takes the viewpoint that indices, not tensors, are the primary owner of dimension information.  So, for example, if we see index `i` or `j` used, we can assume that somewhere, the dimension of `i` and `j` have been defined.

So suppose we have indices `i` and `j` such that `DIMS(i) = [8]` and `DIMS(j) = [9]`.  In the most general case, an index may have multiple dimensions, but for the purposes of this example, both will just have one dimension.

Here are implementations of *index unrolling* and *array mapping*:

```python
import numpy as np

# evaluate index_expr across the cartesian product of index values
# return the evaluations as an array
def unroll(index_expr, dims):
    ufunc = np.frompyfunc(index_expr, nin=len(dims), nout=1)
    return np.fromfunction(ufunc, dims, dtype=int)

# create a new array mapping each ind in unrolled_inds to its value in ary
# returning 0 for out of bounds indices
def map_ary(unrolled_inds, ary):
    def pyfunc(idx):
        return ary[idx] if idx in np.ndindex(ary.shape) else 0
    ufunc = np.vectorize(pyfunc)
    return ufunc(unrolled_inds)
```

Unrolling an index expression produces an array, with dimensions equal to the set of index variables in the expression.  By convention, index expressions produce integer tuple values, even if just one element.

```python
dims_i = 8
dims_j = 9
# index expression is '(i,)' which could appear as ary1d[i]
print(unroll(lambda i: (i,), (dims_i,)))
[(0,) (1,) (2,) (3,) (4,) (5,) (6,) (7,)]

# index expression is '(j,)' which could appear as ary1d[j]
print(unroll(lambda j: (j,), (dims_j,)))
[(0,) (1,) (2,) (3,) (4,) (5,) (6,) (7,) (8,)]

# index expression is '(i,j)' which could appear as ary2d[i,j]
print(unroll(lambda i,j: (i,j), (dims_i, dims_j)))
[[(0, 0) (0, 1) (0, 2) (0, 3) (0, 4) (0, 5) (0, 6) (0, 7) (0, 8)]
 [(1, 0) (1, 1) (1, 2) (1, 3) (1, 4) (1, 5) (1, 6) (1, 7) (1, 8)]
 [(2, 0) (2, 1) (2, 2) (2, 3) (2, 4) (2, 5) (2, 6) (2, 7) (2, 8)]
 [(3, 0) (3, 1) (3, 2) (3, 3) (3, 4) (3, 5) (3, 6) (3, 7) (3, 8)]
 [(4, 0) (4, 1) (4, 2) (4, 3) (4, 4) (4, 5) (4, 6) (4, 7) (4, 8)]
 [(5, 0) (5, 1) (5, 2) (5, 3) (5, 4) (5, 5) (5, 6) (5, 7) (5, 8)]
 [(6, 0) (6, 1) (6, 2) (6, 3) (6, 4) (6, 5) (6, 6) (6, 7) (6, 8)]
 [(7, 0) (7, 1) (7, 2) (7, 3) (7, 4) (7, 5) (7, 6) (7, 7) (7, 8)]]

# index expression is '(j-i+2,)' which could appear as ary1d[j-i+2]
# note that in this case, the unrolled array has dimensions DIMS(i,j)
# but the element values are one-element tuples.
print(unroll(lambda i,j: (j-i+2,), (dims_i, dims_j)))
[[(2,) (3,) (4,) (5,) (6,) (7,) (8,) (9,) (10,)]
 [(1,) (2,) (3,) (4,) (5,) (6,) (7,) (8,) (9,)]
 [(0,) (1,) (2,) (3,) (4,) (5,) (6,) (7,) (8,)]
 [(-1,) (0,) (1,) (2,) (3,) (4,) (5,) (6,) (7,)]
 [(-2,) (-1,) (0,) (1,) (2,) (3,) (4,) (5,) (6,)]
 [(-3,) (-2,) (-1,) (0,) (1,) (2,) (3,) (4,) (5,)]
 [(-4,) (-3,) (-2,) (-1,) (0,) (1,) (2,) (3,) (4,)]
 [(-5,) (-4,) (-3,) (-2,) (-1,) (0,) (1,) (2,) (3,)]]
```

Now, with these unrolled index expressions, we can apply them to array expressions.  In the example below, the index expression is `(i-1,j-2)`, which has two index variables `i` and `j`, each with one dimension (as defined above).  Therefore, the unrolled expression has two dimensions.  Also note that the value of the expression is a 2-component tuple.  So, we'll apply the unrolled indices to a 2-D array called `grid`, which could be convolutional filter weights, for example.

```python
grid = np.array([
        [1,1,2,1,1],
        [1,3,3,4,4],
        [2,3,9,6,5],
        [1,6,4,4,3],
        [1,1,3,1,1]
])

# unrolling the index expression '(i-1,j-2)'
offset_inds = unroll(lambda i,j: (i-1,j-2), (dims_i, dims_j))
print(offset_inds)
[[(-1,-2) (-1,-1) (-1,0) (-1,1) (-1,2) (-1,3) (-1,4) (-1,5) (-1,6)]
 [(0, -2) (0, -1) (0, 0) (0, 1) (0, 2) (0, 3) (0, 4) (0, 5) (0, 6)]
 [(1, -2) (1, -1) (1, 0) (1, 1) (1, 2) (1, 3) (1, 4) (1, 5) (1, 6)]
 [(2, -2) (2, -1) (2, 0) (2, 1) (2, 2) (2, 3) (2, 4) (2, 5) (2, 6)]
 [(3, -2) (3, -1) (3, 0) (3, 1) (3, 2) (3, 3) (3, 4) (3, 5) (3, 6)]
 [(4, -2) (4, -1) (4, 0) (4, 1) (4, 2) (4, 3) (4, 4) (4, 5) (4, 6)]
 [(5, -2) (5, -1) (5, 0) (5, 1) (5, 2) (5, 3) (5, 4) (5, 5) (5, 6)]
 [(6, -2) (6, -1) (6, 0) (6, 1) (6, 2) (6, 3) (6, 4) (6, 5) (6, 6)]]

# evaluating the expression 'grid[i-1,j-2]'
vals = map_ary(offset_inds, grid)
print(vals)
[[0 0 0 0 0 0 0 0]
 [0 0 1 1 2 1 1 0]
 [0 0 1 3 3 4 4 0]
 [0 0 2 3 9 6 5 0]
 [0 0 1 6 4 4 3 0]
 [0 0 1 1 3 1 1 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]]
```

In the above example, both the unrolled index expression and the target array were 2-D.  But in general there need be no relation between the dimension of the index expression and the number of components in its value.

In this example, we evaluate `source[j-i+2]`.  The index expression `j-i+2` still has two indices `i` and `j`, but its value is one component, so it must be applied to a 1-D array.

```python
# unrolling the index expression 'j-i+2'
inds = unroll(lambda i,j: (j-i+2,), (dims_i, dims_j))
print(inds)
[[(2,) (3,) (4,) (5,) (6,) (7,) (8,) (9,)]
 [(1,) (2,) (3,) (4,) (5,) (6,) (7,) (8,)]
 [(0,) (1,) (2,) (3,) (4,) (5,) (6,) (7,)]
 [(-1,) (0,) (1,) (2,) (3,) (4,) (5,) (6,)]
 [(-2,) (-1,) (0,) (1,) (2,) (3,) (4,) (5,)]
 [(-3,) (-2,) (-1,) (0,) (1,) (2,) (3,) (4,)]
 [(-4,) (-3,) (-2,) (-1,) (0,) (1,) (2,) (3,)]
 [(-5,) (-4,) (-3,) (-2,) (-1,) (0,) (1,) (2,)]]

source = np.array([1,4,7,5,2])

# evaluate source[j-i+2]
vals = map_ary(inds, source)
print(vals)
[[7 5 2 0 0 0 0 0]
 [4 7 5 2 0 0 0 0]
 [1 4 7 5 2 0 0 0]
 [0 1 4 7 5 2 0 0]
 [0 0 1 4 7 5 2 0]
 [0 0 0 1 4 7 5 2]
 [0 0 0 0 1 4 7 5]
 [0 0 0 0 0 1 4 7]]
```

Notice now that the mapping result takes the same shape as the unrolled indices, and the `source` array values are duplicated.  You might also notice that this looks like the positions of filter weights moved across some input, and indeed this is the expression used in the [first formulation](https://github.com/hrbigelow/einsum-tuple/blob/master/ops/conv_valid_v1.et) of convolution, in which the filter is unrolled along the output spatial dimension.


```python
# Convolution performed by unrolling the filter across the output, using index expression ipos-opos
output[batch,opos,ochan] = filters[ipos-opos,ichan,ochan] * input[batch,ipos,ichan]
```

Of course, this is not a memory efficient way to compute convolution, but it provides a simple definition.  Looking at the formula, it is also easy to see that the `input` tensor appears alone as a single multiplicative term.  Therefore the operation is easily shown to be linear in the input.

