"""
Miscellaneous Module.
"""
import time
import matplotlib.pyplot as plt

def time_module(module, *args, **kwargs):
    """
    Time the run time of a module.

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


def save_all_figures(full_directory,
                     axis_equal=True, tight=True,
                     extension='eps', rcparams=None):

    if rcparams is not None:
        plt.rc_context(rcparams)

    # Go through each figure and save them
    for i in plt.get_fignums():
        fig = plt.figure(i)
        if axis_equal:
            plt.gca().set_aspect('equal', adjustable='box')
        if tight:
            fig.tight_layout()
        fig.savefig('%sfigure%d.%s' % (full_directory, i, extension))
    print('figures saved.')