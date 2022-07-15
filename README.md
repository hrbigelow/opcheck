
$\newcommand\ein[1]{ \color{gray}{\mathsf{#1}} }$
$\newcommand\aryid[1]{ \color{cornflowerblue}{#1} }$
$\newcommand\sym[1]{ \color{black}{#1} }$
$\newcommand\com[0]{ \sym{,\,} }$
$\newcommand\ary[2]{ \ein{ \aryid{#1} \sym{[} #2 \sym{]} } }$
$\newcommand\rankid[0]{ \color{firebrick}{RANK} }$
$\newcommand\dimsid[0]{ \color{firebrick}{DIMS} }$
$\newcommand\func[2]{ \ein{ \color{black}{#1} } \sym{(} #2 \sym{)} }$
$\newcommand{\flatid}[0]{FLAT}$
$\newcommand\rank[1]{ \ein{ \rankid \sym{(} #1 \sym{)} } }$
$\newcommand\dims[1]{ \ein{ \dimsid \sym{(} #1 \sym{)} } }$
$\newcommand{\flatcall}[1]{ \ein{\flatid \sym{(} #1 \sym{)} } }$


# Einsum Tuple - a mini-language for defining tensor operations

Einsum Tuple language is an extension of Einstein Summation (einsum) notation, with these rules.

1. indices are tuples of unspecified length
2. tensors can be indexed with arbitrary expressions of indices (*index expressions*)
3. out-of-bounds index values are silently ignored
4. like einsum notation, broadcasting and summation are automatic
5. unlike einsum notation, indexing expressions appear in brackets, not subscripts

For example, in the statement:

$
\begin{aligned}
\ary{output}{batch \com row \com col} & = 
\ary{input}{batch \com row \com inner} \sym{*}
\ary{weights}{batch \com inner \com col}
\end{aligned}
$

the identifiers $\ein{batch}$, $\ein{row}$, $\ein{col}$, and $\ein{inner}$ are "eintups", the Einsum Tuple equivalent of Einstein indices.  Eintups symbolize a tuple of an unspecified number (even zero) of individual indices.  For instance:

$
\begin{array}{ll}
\ein{\aryid{output}}_{brc} & = \ein{\aryid{input}}_{bri} \ein{\aryid{weights}}_{bic} \\ 
\ein{\aryid{output}}_{b_{1}b_{2}rc} & = \ein{\aryid{input}}_{b_{1}b_{2}ri} \ein{\aryid{weights}}_{b_{1}b_{2}ic} \\ 
\ein{\aryid{output}}_{b_{1}b_{2}b_{3}rc} & = \ein{\aryid{input}}_{b_{1}b_{2}b_{3}ri} \ein{\aryid{weights}}_{b_{1}b_{2}b_{3}ic} \\ 
\ein{\aryid{output}}_{br_{1}r_{2}c} & = \ein{\aryid{input}}_{br_{1}r_{2}i} \ein{\aryid{weights}}_{bic} \\ 
...
\end{array}
$



## Rank and Dimensions 

Before execution, the runtime system configures shapes of all eintups.  It does this in two phases, first choosing a *rank*, or number of indices, and then generates *dimensions*.  For example, the system might set the rank of batch to 3 and its dimensions to `[2,4,3]` before execution.  These quantities can be accessed in the language as $\dims{batch}$ and $\rank{batch}$.  They are constants during the execution phase, but change at each new shape configuration.

After shape configuration, the shapes of every array expression is known.  For example, the shape of $\ein{\aryid{output}}$  would be $\dims{batch \com row \com col}$, which is shorthand for $\dims{batch}$ concatenated with $\dims{row}$ and then $\dims{col}$.  Its rank is given by $\rank{batch \com row \com col}$ .

## Index expressions

Index expressions are arithmetic expressions of eintups and integer valued arguments, and sometimes functions of them.  For example, here is the `tf.nn.convolution` operation expressed in Einsum Tuple, using the `padding=VALID` option.

$
\begin{aligned}
\ary{output}{batch \com opos \com ochan} \sym{=\,} & 
\ary{filters}{ipos \sym{-} \dims{stride} \sym{*} opos \com ichan \com ochan} \\
\sym{*\,} & \ary{input}{batch \com ipos \com ichan}
\end{aligned}
$


The index expression is $\ein{ipos} \sym{-} \dims{stride} \sym{*} \ein{opos}$.  To be well-formed, binary operation between eintups and $\dims{\cdots}$ arguments must have equal rank.  In this case, $\ein{ipos}$, $\ein{stride}$, and $\ein{opos}$ must have the same rank or the statement won't compile.

Note that this expression can have component values that are negative.  Einsum Tuple implicitly ignores out-of-bounds indices (negative or too high).  If any component of an index expression in the top level assignment statement is out of bounds, the whole calculation doesn't participate in the Einstein summation / assignment.

In the statement above, it is clear that the convolution operation is linear in the $\ein{\aryid{input}}$ tensor since it appears with a single multiplicative term.  An equivalent formula showing linearity in the $\ein{\aryid{filters}}$ is given by:

$
\begin{aligned}
\ary{output}{batch \com opos \com ochan} & \sym{=}
\ary{filters}{fpos \com ichan \com ochan} \\
& \sym{*} \ary{input}{batch \com fpos \sym{+} \dims{stride} \sym{*} opos \com ichan}
\end{aligned}
$

## Indexing expression Basis

The indexing expression $\ein{ipos - \dims{stride} * opos}$ can be thought of as a computed tensor over the space $\dims{ipos \com opos}$  whose elements are 1D tuples with $\rank{ipos}$ members.  Each element of this virtual tensor is then used to index into its parent array $\ein{\aryid{filters}}$ , which expects a tuple of that size.

The set $\ein{ipos \com opos}$ of eintups is known as the *basis* for the indexing expression, and it is derived as the set of all eintup variables in the expression.  Note that while $\ein{stride}$ appears, it isn't a variable because $\dims{stride}$ resolves to a constant at runtime.

The basis of the full index list in the expression $\ary{filters}{ipos \sym{-} \dims{stride} \sym{*} opos \com ichan \com ochan}$ is then $\ein{ipos \com opos \com ichan \com ochan}$.  This is one eintup larger than the basis of the $\ein{\aryid{filters}}$ array to begin with.  Thus, one can think of this as a sort of calculated broadcast, or 'unrolling' of the filters tensor but in a diagonal direction.  The convolution becomes a fully connected layer "matrix multiplication" using this unrolled filter matrix.

In the second form, the input is 'unrolled' and becomes a matrix which multiplies the filters.

## Implicit Broadcasting

Einsup Tuple statements perform implicit broadcasting of tuples which appear on the left hand side of an assignment but not on the right.  For example, here is a formula for `tf.meshgrid`, which constructs a set of broadcasted tensors in a coordinated way.

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

The expression $\ein{ipos \sym{//} \dims{bsz}}$ takes on distinct values in the pattern of 'non-overlapping blocks` of the input locations.

> The Y, X coordinates within each block of the input become the high order component of the output channel index.

The expression $\ein{ipos \sym{\%\,} \dims{bsz}}$ is the 'Y, X coordinates within each block', and it appears as the high order component in the overall expression $\func{FLAT}{\ein{ipos} \sym{\%\,} \dims{bsz} \com \ein{ichan}}$.

> The depth of the output tensor is block_size * block_size * input_depth.

The expression $\ein{ipos \sym{\%\,} \dims{bsz}}$ takes on values up to the exclusive range $\dims{bsz}$.  `input_depth` corresponds to $\ein{ichan}$ and finally, the $\func{FLAT}{}$ call creates values in the range of the product of dimensions.

# Using an Array as Index Expression

An Eintup array of $n$ dimensions with integer elements may be viewed as a an array of $n-1$ dimensions whose elements are 1D integer tuples of a fixed size.  For example, integer valued array $\ary{indices}{batch \com coord \com elem}$ of shapes $\dims{batch} = [3,5,7,4,2]$ as a 7-element integer tuple array of shape $[3,5,4,2]$.  

$
\begin{aligned}
\ary{indices}{slice \com coord} & = \func{RANDOM}{0, 10, \mathsf{INT}} \\
\ary{output}{\ary{indices}{slice \com \sym{:}} \com elem} & = \cdots 
\end{aligned}
$

The index expression is $\ary{indices}{slice \com \sym{:}}$ .  It is like an ordinary array access, except that exactly one index has been called with $\sym{:}$.  In this case, the index called with $\sym{:}$ was $\ein{coord}$.  In order to be legal, $\rank{coord}$ must equal 1, and $\dims{coord}$ must equal $[\rank{usage}]$ where $\ein{usage}$ is the signature where it is used.

Using an array slice as an index expression is a scatter operation if used on the left hand side, and a gather operation if on the right.

# Runtime


The .et file third section is the 'TF Call'.  It is a quasi-Python function call statement which is preprocessed in a few ways:

1. Bare identifiers resolve to tensors mentioned in the Einsup Tuple program
2. $\dims{\cdots}$ expressions resolve to a Python list of integers
3. $\rank{\cdots}$ expressions resolve to a Python integer
4. The $\func{L}{\cdots}$ function allows passing in a Python literal
5. The $\func{TENSOR}{\cdots}$ function creates a tensor from $\dims{}$, $\rank{}$ or integer arguments.

The runtime will run the TF Call, and then compare the output(s) with those listed on the Outputs line, which are computed using the Einsum Tuple program.

# Constraints



# Einsum Tuple Full Grammar Specification

The full grammar specification can be found in eintup_grammar.ebnf.

