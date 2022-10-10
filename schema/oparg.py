"""
Subclasses of OpArg - a class for representing arguments to the op, which are
returned by certain nodes of gen_graph
"""
import numpy as np
import tensorflow as tf
from .error import SchemaError

class OpArg(object):
    def __init__(self, *args):
        pass

    def value(self):
        """
        Generate the value for use by the framework op
        """
        raise NotImplementedError

    def summary(self):
        """
        Return a short string summary of the argument
        """
        raise NotImplementedError

class DataTensorArg(OpArg):
    """
    An OpArg produced by ge.DataTensor 
    """
    def __init__(self, shape, dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def value(self):
        nelem = np.prod(self.shape)
        if nelem > 1e8:
            raise SchemaError(f'Shape \'{self.shape}\' has {nelem} elements, '
            f'which exceeds 1e8 elements')
        if self.dtype.is_integer:
            lo = max(self.dtype.min, -1000)
            hi = min(self.dtype.max, 1000) 
            ten = tf.random.uniform(self.shape, lo, hi, dtype=tf.int64)
            ten = tf.cast(ten, self.dtype)
        elif self.dtype.is_floating:
            lo = max(self.dtype.min, -1.0)
            hi = min(self.dtype.max, 1.0)
            ten = tf.random.uniform(self.shape, lo, hi, dtype=tf.float64)
            ten = tf.cast(ten, self.dtype)
        elif self.dtype.is_bool:
            ten = tf.random.uniform(self.shape, 0, 2, dtype=tf.int32)
            ten = tf.cast(ten, self.dtype)
        elif self.dtype.is_quantized:
            lo, hi = -1000, 1000
            ten = tf.random.uniform(self.shape, lo, hi, dtype=tf.float32)
            quant = tf.quantization.quantize(ten, lo, hi, self.dtype)
            ten = quant.output
        elif self.dtype.is_complex:
            lo, hi = -1.0, 1.0
            real = tf.random.uniform(self.shape, lo, hi, dtype=tf.float64)
            imag = tf.random.uniform(self.shape, lo, hi, dtype=tf.float64)
            ten = tf.complex(real, imag, self.dtype)
        else:
            raise SchemaError(
                f'Unexpected dtype when generating tensor: dtype=\'{self.dtype.name}\'')
        return ten

    def summary(self):
        return f'DTen({self.shape}:{self.dtype.name})'

class ShapeTensorArg(OpArg):
    """
    An OpArg produced by ge.ShapeTensor
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def value(self):
        return tf.constant(self.shape, dtype=tf.int32)
    
    def summary(self):
        return f'ShTen({self.shape})'

class ShapeListArg(OpArg):
    """
    An OpArg produced by ge.ShapeList
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def value(self):
        return self.shape

    def summary(self):
        return f'L({self.shape})'

class ShapeTensor2DArg(OpArg):
    """
    An OpArg produced by ge.ShapeTensor2D
    """
    def __init__(self, shape2d):
        self.content = shape2d

    def value(self):
        ten = tf.constant(self.content, dtype=tf.int32)
        ten = tf.transpose(ten)
        return ten

    def summary(self):
        content = ', '.join(str(r) for r in self.content)
        return f'Sh2Ten({content})'

class ShapeIntArg(OpArg):
    """
    An OpArg produced by ge.ShapeInt
    """
    def __init__(self, val):
        self.val = val

    def value(self):
        return self.val

    def summary(self):
        return f'I:{self.val}' 

