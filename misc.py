import tensorflow as tf
from parse import BCParser

def validate(cfg, json_entry):
    dat = json_entry['tfcall']
    parser = BCParser(cfg)
    parser.set_argument_mode()

    if isinstance(dat['args'], list):
        args = [ parser.parse(arg).value() for arg in dat['args'] ]
        kwargs = { }
    elif isinstance(dat['args'], dict):
        args = [ ]
        kwargs = { k: parser.parse(v).value() for k, v in dat['args'].items() }
    else:
        dat_args_type = type(dat['args'])
        raise RuntimeError(
            f'expected list or object JSON for tfcall "args" field. '
            f'Got {dat_args_type}')

    if 'const-args' in dat:
        if isinstance(dat['const-args'], list):
            args.extend(dat['const-args'])
        elif isinstance(dat['const-args'], dict): 
            kwargs.update(dat['const-args'])
        else:
            const_args_type = type(dat['const-args'])
            raise RuntimeError(
                f'expected list or object JSON for tfcall "const-args" field. '
                f'Got {const_args_type}')

    func = eval(dat['func'])
    tf_results = func(*args, **kwargs)

    return_tensors = dat['return-value']
    if isinstance(return_tensors, str):
        return_tensors = [return_tensors]
    if not isinstance(tf_results, (list, tuple)):
        tf_results = [tf_results]

    et_results = [ parser.parse(name).value() for name in return_tensors ]

    if len(tf_results) != len(et_results):
        raise RuntimeError(
            f'Got {len(tf_results)} results from TF native call '
            f'but {len(et_results)} from eintup program')

    z = zip(tf_results, et_results)
    equal = [equal_tensors(tf_res, et_res, 1e-6) for tf_res, et_res in z]
    return equal

    # print(cfg.array_sig[return_tensor])
    # print(tf.reduce_sum(tf.subtract(tf_result, eintup_result)))
    # print(tf_result.device)


