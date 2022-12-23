import multiprocessing
# multiprocessing.set_start_method('spawn')

from multiprocessing import Process, Queue, Pipe

def target(sub):
    try:
        op = sub.recv()
        op_args = sub.recv()
        print('in target:', op_args, flush=True)
        arg_dict = { k: v.value() for k, v in op_args.items() }
        # ignore any return value
        op.wrapped_op(**arg_dict)
    except BaseException as ex:
        sub.send(ex)
        # send.send('hi')
    except:
        print('some other exception', flush=True)
    finally:
        sub.send(None)
        sub.close()

def proc_wrap(op, op_args):
    par, sub = Pipe(duplex=True)

    # op_args = { 'data_format': 'NCW' }
    p = Process(target=target, args=(op, sub))
    par.send(op)
    par.send(op_args)
    p.start()
    p.join()
    # print('got here in proc_wrap', flush=True)
    exc = sub.recv()
    # exc = None
    # recv.close()
    if exc is not None:
        raise exc 
    if p.exitcode != 0:
        raise RuntimeError(f'got exit code {p.exitcode}')


