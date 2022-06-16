# Tuple Einstein Notation Mini-language

Here I introduce a generalization to Einstein notation expressions.  Such
an expression is useful for defining a Tensorflow operation which generalizes
across multiple combinations of dimensions.

## Definition of Tuple Einstein notation

Tuple Einstein notation is syntactic sugar for traditional Einstein Notation in
which a group of indices in a certain order is symbolized by a single letter.
For example, a traditional batched sum operation for two batch dimensions and
three trailing dimensions could be expressed as:

```python
# Einstein notation
result[p,q] = input[p,q,r,s,t]
```

which can be written succinctly as:

```python
# Tuple-Einstein notation
# b := (p,q), the batch index
# s := (r,s,t), the summation index
result[b] = input[b,s]
```

Additionally, we introduce the following definitions:

1. `length(s)` is the number of underlying Einstein indices symbolized by `s`
2. `size("s", i)` is the total number of values for the `i`th (zero-based) entry in `s`.
3. `randint(low, hi)` generates a random integer in `[low, hi)`
4. `random()` generates a uniform random floating point number

In the expressions, the first appearance of any array defines its dimensions.

Using the above, here is a full description of all possible `gatherNd` calls
(not including the logic of out-of-bounds indices, however):

```bash
# gatherNd definition
index[b,s,c] = randint(0, size("e",c))'
params[b,e] = random()
result[b,s] = params[b,index[b,s,:]]

# gatherNd dimension constraints in TensorFlow
length(b) + length(s) <= 7
length(b) + length(e) <= 7
length(c) = 1
size("c", 0) = length(s)
```

in which `b` selects a member of the *batch*, `e` selects elements within a
slice, `s` selects a *slice*, and `c` selects a *coordinate*.  Expressions like
`index[b,s,c]` are array slice expressions.  The first such expression in which
the array `index` occurs defines the shape (and thus the rank) of the array.
All following expressions must use the same number of `eintuple` identifiers or
the special symbol `:`, indicating a wildcard.  The wildcard
Once the dimensions and sizes of the indices are defined, the assignments can
be evaluated.

