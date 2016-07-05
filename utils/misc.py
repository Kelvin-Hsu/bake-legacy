import time

def time_module(module, *args, **kwargs):

    t_start = time.clock()
    output = module(*args, **kwargs)
    t_finish = time.clock()
    t_module = t_finish - t_start
    print('Module %s finished in: %f seconds' % (module.__name__, t_module))
    return output, t_module