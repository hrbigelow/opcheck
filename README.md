# Tuple Einstein Notation Mini-language

Here I introduce a generalization to Einstein notation expressions.  Such
an expression is useful for defining a Tensorflow operation which generalizes
across multiple combinations of dimensions.

For example, `gatherNd` is defined as follows:

```bash
index[b,s,c] = randint(0, size("s",c))
params[b,s] = random()
result[b,s] = params[b,index[b,s,:]]
```

## Definition of Tuple Einstein notation

Tuple Einstein notation is syntactic sugar for traditional Einstein Notation in
which a group of indices is symbolized by a single letter.  For example, a
traditional batched sum operation for two batch dimensions and three trailing
dimensions could be expressed as:

```python
# Einstein notation
result[a,b] = input[a,b,c,d,e]
```

can be written succinctly as:

```python
# Tuple-Einstein notation
# b := (a,b), the batch index
# s := (c,d,e), the summation index
result[b] = input[b,s]
```


