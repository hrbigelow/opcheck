import opcheck
import tensorflow as tf

opcheck.init()
opcheck.validate_schema(True)

def test_conv1():
    batch = [10]
    input_spatial = [100, 100]
    filter_spatial = [9, 9]
    input_channel = [4]
    output_channel = [8]

    input = tf.random.normal(batch + input_spatial + output_channel)
    filters = tf.random.normal(filter_spatial + input_channel + output_channel)
    output = tf.nn.convolution(input, filters, padding='VALID')

def test_gather():
    batch = []
    read_loc = [10,20,10]
    # read_loc = []
    write_loc = [15,4,9]
    slice_elem = [5,7]
    component = [3]

    params = tf.random.normal(batch + read_loc + slice_elem)
    indices = tf.random.uniform(batch + write_loc + component, -5, 30, dtype=tf.int32)
    output = tf.gather_nd(params, indices, batch_dims=len(batch))

if __name__ == '__main__':
    # test_conv1()
    test_gather()


