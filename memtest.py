import logging
import io, os
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from opschema import runner
from opschema.error import OpSchemaInternalError
from opschema.redirect import stderr_redirector
import inspect

import opschema
opschema.register('tf.nn.convolution')
op = opschema.REGISTRY['tf.nn.convolution']

if __name__ == '__main__':

    configs = op._generate_tests() 
    tests = []
    for test_id, (stat, args, ranks) in enumerate(configs, 1):
        t = runner.TestResult(op, test_id, args, ranks, stat) 
        tests.append(t)

    for t in tests:
        string_err = io.BytesIO()
        arg_dict = t.make_args()

        try:
            with stderr_redirector(string_err):
                val = op.wrapped_op(**arg_dict)
            # print(f'num return elems: {np.prod(val.shape)}', flush=True)
        except OpSchemaInternalError as e:
            print(string_err.getvalue().decode('UTF-8'))
            raise e
        except BaseException as e:
            # this should always be from TensorFlow
            trace = inspect.trace()
            for frame in reversed(trace):
                mod = inspect.getmodule(frame[0])
                if mod is not None:
                    break
            modname = mod.__name__
            # print(modname, flush=True)
            if modname.split('.')[0] == 'tensorflow':
                # print('exception inside tensorflow:')
                # traceback.print_stack()
                pass
            else:
                assert False, 'exception outside tf should not be possible'
                print('exception outside of tensorflow')
                traceback.print_stack()
                raise e
        t.opschema_errors = t.op.input_errors
        if t.op.framework_error is None:
            t.framework_error = t.op.framework_error
        else:
            t.framework_error = str(t.op.framework_error.ex)

        """
        if len(t.opschema_errors) == 0:
            top_hit = ['No Hit found']
        else:
            top_hit = t.opschema_errors[0]
        if top_hit != t.gen_errors:
            cat = 'FAIL'
        else:
            fr_neg = (t.framework_error is None)
            if len(t.gen_errors) == 0:
                cat = 'TN' if fr_neg else 'FN'
            else:
                cat = 'FP' if fr_neg else 'TP'
        t.category = cat
        """

        mem = tf.config.experimental.get_memory_info('GPU:0')
        print(t.id, mem)


