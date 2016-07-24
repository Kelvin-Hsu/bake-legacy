import numpy as np
import bake
import utils

def main():

    def objective(A, b, x):
        return 0.5 * np.dot(x, np.dot(A, x)) + np.dot(b, x)

    n = 100
    A, b = utils.data.generate_test_quadratic_problem(n)
    print(A)
    print(b)
    x_init = 0.5 * np.ones(n)
    x_opt_1 = bake.optimize.solve_positive_constrained_quadratic_iterative(A, b, x_init)
    print(x_opt_1)
    x_opt_2 = bake.linalg.solve_positive_constrained_quadratic(A, b)
    print(x_opt_2)

    print(objective(A, b, x_opt_1))
    print(objective(A, b, x_opt_2))

    print(x_opt_1 - x_opt_2)

    print(np.any(x_opt_1 < 0))
    print(np.any(x_opt_2 < 0))

if __name__ == "__main__":
    utils.misc.time_module(main)