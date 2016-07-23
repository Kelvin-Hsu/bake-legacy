"""
Miscellaneous Module.
"""
import time


def time_module(module, *args, **kwargs):
    """

    Parameters
    ----------
    module : callable
        The module to be timed.
    args : args
        The arguments of the module.
    kwargs : kwargs
        The keyword arguments of the module.

    Returns
    -------
    tuple
        The output(s) of the module and also the time taken to run the module
    """
    t_start = time.clock()
    output = module(*args, **kwargs)
    t_finish = time.clock()
    t_module = t_finish - t_start
    print('Module %s finished in: %f seconds' % (module.__name__, t_module))
    return output, t_module