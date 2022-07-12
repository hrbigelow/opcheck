# Einsum Tuple Mini-language 

A language for defining Tensorflow operations in a rank-agnostic way

## Synopsis

```bash
# Run an op defined in Einsum Tuple language (.et) on all allowed sub-rank
# combinations of inputs.  Check for validity against the Tensorflow op output.

python eintup.py ops/gather_nd.et
python eintup.py ops/scatter_nd.et
python eintup.py ops/conv_valid_v1.et
python eintup.py ops/tconv_valid_v1.et
python eintup.py ops/matmul.et
...
```

## Examples

The following are complete Einsum Tuple programs for various operations.  In
all examples, the first two lines define the shapes of inputs, and the last
line defines the operation.  The definitions are valid for multiple
combinations of ranks, such as for batch, ipos, ichan, etc.

```
# Strided Convolution (Valid) 
# Expression shows linearity with respect to input
input[batch,ipos,ichan] = RANDOM(0, 10, FLOAT)
filters[fpos,ichan,ochan] = RANDOM(0, 1, FLOAT)
output[batch,opos,ochan] = filters[ipos-DIMS(stride)*opos,ichan,ochan] * input[batch,ipos,ichan]

# Alternate form shows linearity with respect to filters 
output[batch,opos,ochan] = filters[fpos,ichan,ochan] * input[batch,fpos+DIMS(stride)*opos,ichan]

# Transpose convolution, showing linearity with respect to input
input[batch,ipos,ichan] = RANDOM(0, 10, FLOAT)
filters[fpos,ochan,ichan] = RANDOM(0, 1, FLOAT)
output[batch,opos,ochan] = filters[opos-DIMS(stride)*ipos,ochan,ichan] * input[batch,ipos,ichan]

# Alternate form showing linearity with respect to filters
input[batch,ipos,ichan] = RANDOM(0, 10, FLOAT)
filters[fpos,ochan,ichan] = RANDOM(0, 1, FLOAT)
input_aug[batch,apos,ichan] = 0.0
input_aug[batch,DIMS(stride)*ipos,ichan] += input[batch,ipos,ichan]
output[batch,opos,ochan] = filters[fpos,ochan,ichan] * input_aug[batch,opos-fpos,ichan]

# gather_nd
params[batch,elem,other] = RANDOM(0, 10, FLOAT)
indices[batch,slice,coord] = RANDOM(0, DIMS(elem)[coord], INT)
result[batch,slice,other] = params[batch,indices[batch,slice,:],other]

# scatter_nd
indices[slice,coord] = RANDOM(0, DIMS(dest)[coord], INT)
updates[slice,elem] = RANDOM(0, 10, FLOAT)
output[dest,elem] = 0.0 
output[indices[slice,:],elem] = updates[slice,elem]

# meshgrid
in1[a] = RANDOM(0, 10, INT)
in2[b] = RANDOM(0, 10, INT)
in3[c] = RANDOM(0, 10, INT)
in4[d] = RANDOM(0, 20, INT)
out1[a,b,c,d] = in1[a]
out2[a,b,c,d] = in2[b]
out3[a,b,c,d] = in3[c]
out4[a,b,c,d] = in4[d]
```


## Motivation

Many TensorFlow operations work across a combination of ranks for batch,
spatial, indexing and other dimensions.  For instance, `convolution` works for
input with 1, 2, or 3 spatial dimensions.  Batched matrix multiplication works
for a number of batch dimensions from 0 to at least 9.  `gatherNd` works with
variable numbers of dimensions for batching, slice layout and slice
dimensions.

Intuitively, the logic of these operations seems clear, although there is often
not a very clear way to denote it in a rank-agnostic way.  This language
attempts to provide that.

There are several benefits to having such a language:  documentation, unit
test generation, and facilitation of new proposed ops are three possibilities.

## Definition of Tuple Einstein notation

Einsum Tuple notation is inspired by Einstein summation notation, and consists
of tensor assignment statements.  While Einstein summation uses subscripts like
$result_{brc} = mat1_{bri} * mat2_{bic}$, Einsum Tuple notation uses bracketed
indices, like $result[b,r,c] = mat1[b,r,i] * mat2[b,i,c]$.  
Einsum indices denote tuples of Einstein indices in a fixed
order.  In the assignment, indices which appear on the right hand side but
not the left are marginalized out.  Indices which appear on the left but not
the right cause broadcasting of the right-hand-side expression.  



For an example of broadcasting behavior, here is the definition of
`tf.meshgrid` for `N = 4`:

```python
in1[a] = RANDOM(0, 10, INT)
in2[b] = RANDOM(0, 10, INT)
in3[c] = RANDOM(0, 10, INT)
in4[d] = RANDOM(0, 20, INT)
out1[a,b,c,d] = in1[a]
out2[a,b,c,d] = in2[b]
out3[a,b,c,d] = in3[c]
out4[a,b,c,d] = in4[d]

tf.meshgrid(in1, in2, in3, in4, indexing=L('ij'))

out1, out2, out3, out4

RANK(a) = 1
RANK(b) = 1
RANK(c) = 1
RANK(d) = 1
```

The `.et` file has four sections, separated by a blank line.  The first is the
program.  The second is a python-like function call of some TensorFlow op,
which references the tensors defined in the program.  The third section is a
line that names tensors in the program which are to be compared with the return
values of the TensorFlow call.  The last section lists constraints on the ranks
of the Eintups, and also dimensions (not shown in this example).

In the program, four Eintups `a`, `b`, `c`, and `d` are defined, with one array
for each of them.  Then, four output arrays are defined, each with the compound
shape `a,b,c,d`.  The four assignment statements illustrate broadcasting
behavior.  The first one broadcasts along `b,c,d` since these are present
only on the left. 

The above file can be run as:

```bash
python eintup.py ops/meshgrid.et
a     b     c     d     Validated
[50]  [93]  [26]  [90]  [True, True, True, True]
```

The system generates shapes for the four Eintups randomly, obeying the
constraints given.  It then runs the Eintup program, populating the eight
tensors.  It then performs the TensorFlow call.  Finally, it compares the
TensorFlow op outputs with those from the Eintup program, and returns True or
False if equal.

For other examples, see the [ops](ops) directory.  A full description of Einsum
Tuple language is in [eintup.md](eintup.md).

