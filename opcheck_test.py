import opcheck
import tensorflow as tf


def test_conv1():
    batch = [10]
    input_spatial = [100, 100, 100]
    filter_spatial = [9, 9, 9]
    input_channel = [4]
    output_channel = [8]

    # input = tf.random.normal(batch + input_spatial + output_channel)
    input = tf.random.normal(batch + input_spatial + input_channel)
    filters = tf.random.normal(filter_spatial + input_channel + output_channel)
    output = tf.nn.convolution(input, filters, padding='SAME')

def test_gather():
    batch = []
    read_loc = [10,20,10]
    # read_loc = []
    write_loc = [15,4,9]
    # slice_elem = [5,7]
    slice_elem = []
    component = [3]

    params = tf.random.normal(batch + read_loc + slice_elem)
    indices = tf.random.uniform(batch + write_loc + component, -5, 30, dtype=tf.int32)
    output = tf.gather_nd(params, indices, batch_dims=len(batch))

def test_scatter():
    read_address = [10,20,10]
    slice_element = [5]
    dest = [8,15,10] # rank of dest = component[0]
    component = [3]
    output_shape = dest + slice_element
    indices = tf.random.uniform(read_address + component, -5, 30, dtype=tf.int32)
    updates = tf.random.normal(read_address + slice_element)
    output = tf.scatter_nd(indices, updates, shape=output_shape)

if __name__ == '__main__':
    opcheck.init()
    opcheck.validate_schema(True)
    # test_conv1()
    test_gather()
    # test_scatter()

