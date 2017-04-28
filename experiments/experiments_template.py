import numpy as np
import datetime
import os
from tensorflow import name_scope
import cake


def run_experiment(x_train, y_train, x_test, y_test,
                   name='experiment',
                   s_init=1.,
                   fix_s=False,
                   l_init=np.array([1.]),
                   zeta_init=1e-4,
                   learning_rate=0.01,
                   grad_tol=0.01,
                   max_iter=100,
                   n_sgd_batch=None,
                   n_train_limit=5000,
                   objective='full',
                   sequential_batch=False,
                   log_hypers=True,
                   to_train=True,
                   save_step=100):
    """
    Run experiment with the kernel embedding classifier.

    Parameters
    ----------
    x_train : numpy.ndarray
        The training inputs (n_train, d)
    y_train : numpy.ndarray
        The training outputs (n_train, 1)
    x_test : numpy.ndarray
        The test inputs (n_test, d)
    y_test : numpy.ndarray
        The test outputs (n_test, 1)
    name : str, optional
        Name of the experiment

    Returns
    -------
    None
    """
    now = datetime.datetime.now()
    now_string = '_%s_%s_%s_%s_%s_%s' % (now.year, now.month, now.day,
                                        now.hour, now.minute, now.second)
    name += now_string
    print('\n--------------------------------------------------------------\n')
    full_directory = './%s/' % name
    os.mkdir(full_directory)
    tensorboard_directory = full_directory + 'tensorboard/'
    os.mkdir(tensorboard_directory)
    print('Running Experiment: %s' % name)
    print('Results will be saved in "%s"' % full_directory)
    print('Tensorboard results will be saved in "%s"' % tensorboard_directory)

    classes = np.unique(y_train)
    n_class = classes.shape[0]
    print('There are %d classes for this dataset: ' % n_class, classes)
    for c in classes:
        print('There are %d observations for class %d.'
              % (np.sum(y_train == c), c))
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    print('Training on %d examples' % n_train)
    print('Testing on %d examples' % n_test)

    print('Initial Sensitivity: %g' % s_init)
    print('Initial Length Scale: %s' % np.array_str(l_init))
    print('Initial Regularisation Parameter: %g' % zeta_init)
    print('Learning Rate: %g' % learning_rate)
    print('Gradient Error Tolerance: %g' % grad_tol)
    print('Maximum Iterations: %g' % max_iter)
    if n_sgd_batch:
        print('Batch Size for Stochastic Gradient Descent: %g' % n_sgd_batch)
    else:
        print('Using full dataset for Gradient Descent')

    # Specify the kernel and kernel parameters
    if fix_s:
        kernel = lambda *args, **kwargs: \
            (s_init ** 2) * cake.kernels.gaussian(*args, **kwargs)
        theta_init = np.ones(l_init.shape[0])
    else:
        kernel = cake.kernels.s_gaussian
        theta_init = np.ones(l_init.shape[0] + 1)
        theta_init[0] = s_init
        theta_init[1:] = l_init

    # Train the kernel embedding classifier
    with name_scope(name):
        kec = cake.KernelEmbeddingClassifier(kernel=kernel)
        kec.fit(x_train, y_train, x_test, y_test,
                theta=theta_init,
                zeta=zeta_init,
                learning_rate=learning_rate,
                grad_tol=grad_tol,
                max_iter=max_iter,
                n_sgd_batch=n_sgd_batch,
                n_train_limit=n_train_limit,
                objective=objective,
                sequential_batch=sequential_batch,
                log_hypers=log_hypers,
                to_train=to_train,
                save_step=save_step,
                tensorboard_directory=tensorboard_directory)
        result = kec.results()
        kec.sess.close()

    np.savez('%sresults.npz' % full_directory, **result)

    config = {'x_train': x_train,
              'y_train': y_train,
              'x_test': x_test,
              'y_test': y_test,
              'name': name,
              's_init': s_init,
              'fix_s': fix_s,
              'l_init': l_init,
              'zeta_init': zeta_init,
              'learning_rate': learning_rate,
              'grad_tol': grad_tol,
              'max_iter': max_iter,
              'n_sgd_batch': n_sgd_batch,
              'n_train_limit': n_train_limit,
              'objective': objective,
              'sequential_batch': sequential_batch,
              'log_hypers': log_hypers,
              'to_train': to_train,
              'save_step': save_step,
              'tensorboard_directory': tensorboard_directory}

    np.savez('%sconfig.npz' % full_directory, **config)

    f = open('%sresults.txt' % full_directory, 'w')
    f.write('Date and Time: %s\n' % now_string)
    f.write('Experiment: %s\n' % name)
    f.write('Number of classes for this dataset: %d \n' % n_class)
    for c in classes:
        f.write('\tNumber of observations for class %d: %d\n'
                % (c, np.sum(y_train == c)))
    f.write('Size of training data: %d\n' % n_train)
    f.write('Size of test data: %d\n' % n_test)

    if fix_s:
        f.write('Initial Sensitivity (Fixed): %g\n' % s_init)
    else:
        f.write('Initial Sensitivity: %g\n' % s_init)
    f.write('Initial Length Scale: %s\n' % np.array_str(l_init))
    f.write('Initial Regularisation Parameter: %g\n' % zeta_init)
    f.write('Learning Rate: %g\n' % learning_rate)
    f.write('Gradient Error Tolerance: %g\n' % grad_tol)
    f.write('Maximum Iterations: %g\n' % max_iter)
    if n_sgd_batch:
        f.write('Batch Size for Stochastic Gradient Descent: %g\n'
                % n_sgd_batch)
    else:
        f.write('Using full dataset for Gradient Descent\n')
    f.write('----------------------------------------\n')
    f.write('Experiment Configuration:\n')
    config_keys = ['name',
                   's_init',
                   'fix_s',
                   'l_init',
                   'zeta_init',
                   'learning_rate',
                   'grad_tol',
                   'max_iter',
                   'n_sgd_batch',
                   'n_train_limit',
                   'objective',
                   'sequential_batch',
                   'log_hypers',
                   'to_train',
                   'save_step',
                   'tensorboard_directory']
    for key in config_keys:
        quantity = config[key]
        if isinstance(quantity, np.ndarray):
            f.write('%s: %s\n' % (key, np.array_str(quantity, precision=8)))
        elif isinstance(quantity, bool):
            f.write('%s: %s\n' % (key, str(quantity)))
        elif isinstance(quantity, int):
            f.write('%s: %d\n' % (key, quantity))
        else:
            try:
                f.write('%s: %f\n' % (key, quantity))
            except:
                f.write('%s: %s\n' % (key, str(quantity)))
    f.write('----------------------------------------\n')
    f.write('Final Results:\n')
    result_keys = ['theta', 'zeta', 'complexity',
                   'train_acc', 'train_cel', 'train_cel_valid', 'train_msp',
                   'test_acc', 'test_cel', 'test_cel_valid', 'test_msp']
    for key in result_keys:
        quantity = result[key]
        if isinstance(quantity, np.ndarray):
            f.write('%s: %s\n' % (key, np.array_str(quantity, precision=8)))
        else:
            f.write('%s: %f\n' % (key, quantity))
    f.close()
