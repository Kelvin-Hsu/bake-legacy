"""
Compare methods for solving normalized unit constrained quadratic problems.
"""
import numpy as np
import bake
import utils


def main():

    # Setup a quadratic objective
    def objective(a, b, x):
        return 0.5 * np.dot(x, np.dot(a, x)) + np.dot(b, x)

    # Generate some quadratic problem
    n = 250
    a, b = utils.data.generate_test_quadratic_problem(n)

    # Solve for the quadratic problem
    x_init = 0.5 * np.ones(n)
    x_opt_1, _ = utils.misc.time_module(bake.optimize.solve_normalized_unit_constrained_quadratic_iterative, a, b, x_init)
    x_opt_2, _ = utils.misc.time_module(bake.optimize.solve_normalized_unit_constrained_quadratic, a, b, x_init)

    # Clip normalize the solution for comparison
    x_opt_1_clip = bake.infer.clip_normalise(x_opt_1)
    x_opt_2_clip = bake.infer.clip_normalise(x_opt_2)

    # Compare methods
    print("Iterative Method: [Original Objective, Clip-Normalized Objective]", objective(a, b, x_opt_1), objective(a, b, x_opt_1_clip))
    print("Incorporated Method: [Original Objective, Clip-Normalized Objective]", objective(a, b, x_opt_2), objective(a, b, x_opt_2_clip))
    print("Iterative Method Constraint Violation: %s" % ("Bad" if np.any(x_opt_1 < 0) else "Good"))
    print("Incorporated Method Constraint Violation %s: " % ("Bad" if np.any(x_opt_2 < 0) else "Good"))
    print("Iterative Method Input Sum: %f" % np.sum(x_opt_1))
    print("Incorporated Method Input Sum %f: " % np.sum(x_opt_2))


if __name__ == "__main__":
    utils.misc.time_module(main)