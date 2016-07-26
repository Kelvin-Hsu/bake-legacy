import numpy as np
import bake
import utils

def main():
    # STILL UNDER TESTING
    def objective(A, b, x):
        return 0.5 * np.dot(x, np.dot(A, x)) + np.dot(b, x)

    n = 500
    A, b = utils.data.generate_test_quadratic_problem(n)
    print(A)
    print(b)
    x_init = 0.5 * np.ones(n)
    x_opt_1, _ = utils.misc.time_module(bake.optimize.solve_normalized_unit_constrained_quadratic_iterative, A, b, x_init)
    x_opt_2, _ = utils.misc.time_module(bake.optimize.solve_normalized_unit_constrained_quadratic, A, b, x_init)

    x_opt_1_clip = bake.infer.clip_normalise(x_opt_1)
    x_opt_2_clip = bake.infer.clip_normalise(x_opt_2)

    # print(utils.data.joint_data(x_opt_1, x_opt_2))

    print(objective(A, b, x_opt_1), objective(A, b, x_opt_1_clip))
    print(objective(A, b, x_opt_2), objective(A, b, x_opt_2_clip))

    print("Good" if not np.any(x_opt_1 < 0) else "Bad")
    print("Good" if not np.any(x_opt_2 < 0) else "Bad")

    print(np.sum(x_opt_1), np.sum(x_opt_2))

    print(np.arange(x_opt_2.shape[0])[x_opt_2 < 0])
if __name__ == "__main__":
    utils.misc.time_module(main)