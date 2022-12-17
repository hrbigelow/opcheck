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

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def value(self):
        """
        Generate the value for use by the framework op
        """
        raise NotImplementedError

class DataTensorArg(OpArg):
    """
    An OpArg produced by ge.DataTensor 
    """
    def __init__(self, shape, dtype_name):
        super().__init__()
        nelem = np.prod(shape)
        if nelem > int(1e8):
            raise SchemaError(f'Shape \'{shape}\' has {nelem} elements, '
                    f'which exceeds 1e8 elements')
        self.shape = shape
        self.dtype = tf.dtypes.as_dtype(dtype_name)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.shape}:{self.dtype.name})'

    def __str__(self):
        return f'{self.shape}:{self.dtype.name}'

    def value(self):
        try:
            return self._value()
        except BaseException as ex:
            raise SchemaError(
                f'{type(self).__qualname__}: Couldn\'t create value for '
                f'argument with shape \'{self.shape}\' and dtype '
                f'\'{self.dtype.name}\'.  Got exception: '
                f'{ex}')

    def _value(self):
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

class ShapeTensorArg(OpArg):
    """
    An OpArg produced by ge.ShapeTensor
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def value(self):
        return tf.constant(self.shape, dtype=tf.int32)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.shape})'

    def __str__(self):
        return f'{self.shape}'

class ShapeListArg(OpArg):
    """
    An OpArg produced by ge.ShapeList
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'{self.__class__.__name__}({self.shape})'

    def __str__(self):
        return f'{self.shape}'

    def value(self):
        return self.shape

class ShapeTensor2DArg(OpArg):
    """
    An OpArg produced by ge.ShapeTensor2D
    """
    def __init__(self, shape2d):
        self.content = shape2d

    def __repr__(self):
        content = ', '.join(str(r) for r in self.content)
        return f'{self.__class__.__name__}({content})'

    def __str__(self):
        content = ', '.join(str(r) for r in self.content)
        return content

    def value(self):
        ten = tf.constant(self.content, dtype=tf.int32)
        ten = tf.transpose(ten)
        return ten

class IntArg(OpArg):
    """
    An OpArg produced by ge.ShapeInt
    """
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return f'{self.__class__.__name__}({self.val})' 

    def __str__(self):
        return f'{self.val}'

    def value(self):
        return self.val

class ValueArg(OpArg):
    """
    An OpArg holding an arbitrary value
    """
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return f'{self.__class__.__name__}({self.val})'

    def __str__(self):
        return f'{self.val}'

    def value(self):
        return self.val

