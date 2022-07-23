# Specification of Einsum Tuple language

The **Einsum Tuple** language consists entirely of array assignment statements of
the forms:

```python
array[index_expr_list] = rval_expr
array[index_expr_list] += rval_expr
```

where `rval_expr` is an arithmetic expression of scalars and array access
expressions.  Each array access (on the left or right) is indexed by a list of
*index expression*s, which are themselves arithmetic expressions of index
variables and integers.  The index variable in **Einsum Tuple** is called the
*eintup*, short for Einstein tuple.  It is a named collection of Einstein
summation indices in a specific order.  The *eintups* play the same role in an
array assignment statement as an Einstein summation expression, but in
multi-dimensional form.  Whereas a simple Einstein index may take on values
`[0, d)`, an *eintup* with k components takes on all values in the rectangular
region `[0, d_1) x [0, d_2) x ...  x [0, d_k)`.

The upper bounds `[d_1, ..., d_k]` are called the *dimensions* of the Eintup.
The dimensions, and the choice of k, are both made by the runtime before
execution.  In this manner, multiple combinations of choices k for different
Eintups can be made.  This allows a single **Einsum Tuple** program to describe
an operation in a *rank agnostic* way.

## Statement Evaluation

Evaluating a statement in **Einsum Tuple** is equivalent to the following
procedure.  It is very similar to that of Einstein summation, except for a few
semantic differences.

1. If the left hand side array name is the first mention of it in the program,
   instantiate the array with shape determined by `index_expr_list` and
   initialize every element with 0
2. Identify all *eintup* variables in the statement (on the left and right)
3. Instantiate the statement with each possible combination of values of the
   *eintup* variables, but with '+=' as the operator
4. For each statement instantiation:
   * If any component of an index is out of bounds (either negative or too
      high), do not execute that instantiation
   * If the statement is of the '=' form and this is the first instantiation
   of a particular setting for the left hand side index list, set the array
   cell to 0 before executing
   * execute the instantiation

There are a few language constructs that are described separately.  But, the
procedure above is a complete description of the execution of all statements.

From these rules alone we can deduce the following higher level properties.

1. *eintups* appearing only on the right hand side are marginalized out
2. *eintups* appearing only on the left hand side are broadcasted into

Note that, unlike Einstein summation, the `rval_expr` may be a single term.
Or, if it is an expression, it need not involve multiplication.

## Dimension and Rank of an Eintup

In the same way that an Einstein summation index takes on values in
some range `[0, d)`, an Eintup with `k` components takes on values in a
hyper-rectangular region `[0, d_1) x [0, d_2), ..., [0, d_k)`.  We then say
that the Eintup 'has dimensions `[d_1, d_2, ..., d_k]` and rank `k`.  The
dimension is defined as the component-wise maximal value plus 1.

More generally, *index expressions* are created as arithmetic expressions of
Eintups and integers (or integer tuples).  Such index expressions all have a
well defined dimension (see **Dimensions of an Index Expression** below).
However, they may not take on all possible values in the implied
hyper-rectangular region.

The dimensions and rank of an Eintup may be accessed at runtime using the
constructs `DIMS(tup)` and `RANK(tup)`.  More generally, the dimensions and
rank for an arbitrary index expression list may be accessed this way using
`DIMS(index_expr_list)` and `RANK(index_expr_list)`.  The dimensions are simply
concatenated from the dimensions of each item in the list, while the rank is
the sum of ranks of the items in the list. 

## Automatic Broadcasting and Automatic Marginalization

All binary operations, including assignment, perform automatic broadcasting of
eintups that are present in just one operand.  For example, the expression

```csharp
outer[batch,row,col] = vec1[batch,row] * vec2[batch,col]
``` 

automatically broadcasts `vec1` across eintup `col` and `vec2` across `row`.
Special to assignment, eintups which exclusively appear on the right of an
assignment will be marginalized out:

```
# day is marginalized out
monthly_totals[batch,month] = daily[batch,month,day]
```

Eintups appearing exclusively on the left of an assignment cause the right side
expression to be broadcasted across those eintups before the assignment.  For
example:

```
# ary is broadcasted across dup before assignment
outer[batch,dup] = ary[batch]
```

## Insensitivity to Relative Index Order in Binary array expressions 

The naming of Eintups allows binary array expressions to work regardless of the
relative order of indices of their operands.  For example, all three
multiplication operations below are equivalent, regardless of the relative
orders of the indices of the left and right sides of the multiplication.

```
mat1[batch,row,inner] = ... # initialization
mat2[inner,col,batch] = ... # initialization

# Define two transposed versions of mat1
m1p1[row,batch,inner] = mat1[batch,row,inner]
m1p2[row,inner,batch] = mat1[batch,row,inner]

m2p1[col,batch,inner] = mat2[inner,col,batch]
m2p2[inner,col,batch] = mat2[inner,col,batch]

# Any combination of (mat1, m1p1, m1p2) can be multiplied by (mat2, m2p1, m2p2)
# Only the combination of *sets* of indices matter, not their order
result[batch,row,col] = mat1[batch,row,inner] * mat2[inner,col,batch]
result[batch,row,col] = m1p1[row,batch,inner] * mat2[inner,col,batch]
result[batch,row,col] = m1p2[row,inner,batch] * mat2[inner,col,batch]
result[batch,row,col] = m1p1[row,batch,inner] * m2p1[col,batch,inner]
result[batch,row,col] = m1p2[row,inner,batch] * m2p2[inner,col,batch]
```

## Index Expressions

Eintups are the simplest kind of expression that can be used within the
brackets of a tensor.  The language accepts arbitrary *index expression*s,
which can be formed as arithmetic binary operations of Eintups and/or
integers.  Using such expressions, one can easily express notions such as
Convolution:

```
input[batch,ipos,ichan] = RANDOM(0, 10, FLOAT)
filters[fpos,ichan,ochan] = RANDOM(0, 1, FLOAT)
output[batch,opos,ochan] = filters[ipos-DIMS(stride)*opos,ichan,ochan] * input[batch,ipos,ichan]

# Alternate form
output[batch,opos,ochan] = filters[fpos,ichan,ochan] * input[batch,fpos+DIMS(stride)*opos,ichan]
```

The *index expression* in the first form of the convolution operation is
`ipos-DIMS(stride)*opos`.  Binary operations between two Eintups are
element-wise, and their ranks must match at runtime.  In the above expression,
the rank of `ipos`, `stride`, and `opos` all must match.

## Dimensions of an Index Expression

An Eintup is the simplest form of an *index expression* in the Einsum Tuple
language.  

Every index expression has a well-determined dimension at runtime.  This is
derived from the dimensions assigned to each Eintup.  When used within the
brackets of an array, the list of index expressions defines the dimension
(shape) of that array.  In the Einsum Tuple language, the first mention of the
array in the program determines its dimensions.

In the same way, the `DIMS(index_expr)` is the component-wise exclusive upper
bound (one more than the max component value) of the Eintup expression.
Calculating this for different expressions is straightforward but sometimes
counterintuitive.  For example:

```python
# This reaches a maximal value when `other` is (0, 0, ..., 0)
DIMS(tup - other) = DIMS(tup)  

# The mod operation may or may not truncate the maximal value, depending
# on the component-wise maxima of `tup` and `other`
DIMS(tup % DIMS(other)) = min(DIMS(tup), DIMS(other))

# Here, static_expr can be an integer or DIMS(other)
# All of these binary expressions follow the same logic.  Since the maximum
# value of tup is one less than DIMS(tup), we take the component-wise
# modified maximum value, and add 1 to get the exclusive upper bound
DIMS(tup // static_expr) = (DIMS(tup) - 1) // static_expr + 1
DIMS(tup //^ static_expr) = (DIMS(tup) - 1) //^ static_expr + 1
DIMS(tup * static_expr) = (DIMS(tup) - 1) * static_expr + 1
```

The full logic of dimension calculation can be found in [ast_nodes.py
SliceBinOp::dims()](https://github.com/hrbigelow/einsum-tuple/blob/40f08f1995af97eb93257d65547e7abb9aa3c9db/ast_nodes.py#L325)

## The Array Slice index expression

*Einsum Tuple* supports an index expression based on the idea of an array (or
tensor) slice.  To introduce it, note that a tensor of k dimensions with
integer elements may be viewed as a tensor of k-1 dimensions whose elements are
1D integer tuples of a fixed size.  For example, let's assume an integer valued
array `indices[slice,coord]`  of shapes `DIMS(slice)=[3,5]`, `DIMS(coord)=[7]`
so that the full shape of `indices` is `[3,5,7]`. We can view the tensor itself
as a function of two arguments (the component values of `DIMS(slice)`) which
outputs 7-tuples of integers. The space over which the two arguments vary are
the dimensions `[3,5]`. Just like an Eintup, the array slice is a set of
tuples, and can be used as such.  Using it as an index expression, we have:

```
indices[slice,coord] = ... # initialization
output[indices[slice,:],elem] = RANDOM(0,10,INT)
```

The index expression is `indices[slice,:]`. It is like an ordinary array
access, except that `coord` has been called with the special ":" symbol. In
order to be legal, `RANK(coord)` must equal 1, and `DIMS(coord)` must equal the
rank of the first place in the `output` array. Note that it would be perfectly
valid if the ":" were in a non-terminal position. For example, using
`indices[coord,slice]` as the array, and `indices[:,slice]` as the index
expression is also valid.

Using an array slice as an index expression is a scatter operation if used on
the left hand side, and a gather operation if on the right.

## Automatic Rank Equality Constraints

The runtime system infers that EinTups have the same rank in the following
situations.  First, if any two EinTups are used together as variables in an
index expression, such as `ipos-DIMS(stride)*opos`.  This will equate the ranks
for `ipos`, `stride`, and `opos`.  Second, if two index expressions are used in
  the same place in the same array, their ranks will be equated.  For instance:

```
trim[dest] = 0
trim[elem-DIMS(left_trim)] = input[elem]
```

Because `dest` and the expression `elem-DIMS(left_trim)` are used in the first
position of array `trim`, the ranks of `dims`, `elem`, and `left_trim` will be
equated.

Finally, when `DIMS()` calls, with a single EinTup argument, are used in a
binary expression in a constraint, their ranks will be equated.  For example:

`DIMS(dest) = DIMS(elem) - DIMS(left_trim) - DIMS(right_trim)`

Because the right hand side is an arithmetic expression using multiple `DIMS()`
calls, the ranks of `elem`, `left_trim`, and `right_trim` are automatically
equated.

In these contexts, 'equated' means that the runtime system automatically
constraints them to have the same ranks when generating ranks and dimensions.

## An Einsum Tuple Array is sized on first use

Einsum Tuple defines the shape of an Array at the moment it first appears in
the program, which must be on the left hand side of an assignment.  The shape
is determined by the dimensions of each member of the index expression list.
Subsequent usage of the array (on the left or right) is only restricted by the
runtime in two ways.  First, that the number of index expressions used to index
it must match that in the first use.  Although one might imagine having a
rank-3 Eintup and trying to use it in place of a rank-1 and rank-2 Eintup, this
is not allowed.  Second, the index expression ranks must match those of the
first use.  

For example, here is the Einsum Tuple definition of `tf.scatter_nd`.  The Array
`output` first appears on line 3, with Eintups `dest`, `elem`.  At runtime,
when the execution reaches line 3, the shape of `output` will be determined by
the dimensions of `dest` and `elem` which are initialized before execution.  On
the fourth line, the array slice index expression `indices[slice,:]` is used in
place of `dest`.  The system will check that it has the same rank as `dest`.

```
indices[slice,coord] = RANDOM(0, DIMS(dest)[coord], INT)
updates[slice,elem] = RANDOM(0, 10, FLOAT)
output[dest,elem] = 0.0 
output[indices[slice,:],elem] = updates[slice,elem]
```

