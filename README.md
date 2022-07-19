
# Einsum Tuple - a mini-language for defining tensor operations

Einsum Tuple language is an extension of Einstein Summation (einsum) notation, with these rules.

1. indices are tuples of unspecified length
2. tensors can be indexed with arbitrary expressions of indices (*index expressions*)
3. out-of-bounds index values are silently ignored
4. like einsum notation, broadcasting and summation are automatic
5. unlike einsum notation, indexing expressions appear in brackets, not subscripts

For quick examples of popular tensor operations defined in Einsum Tuple, see
the [ops directory](https://github.com/hrbigelow/einsum-tuple/tree/master/ops).
For the main documentation, see this
[article](https://www.mlcrumbs.com/einsum-tuple/public/index.html).

