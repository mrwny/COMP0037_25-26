#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
import time
import numpy as np

# Computes Maximum Absolute Difference
def max_abs_diff_vfunc(airport_map, v, vBase,):
    max_diff = 0.0
    for x in range(airport_map.width()):
        for y in range(airport_map.height()):
            diff = abs(v.value(x, y) - vBase.value(x, y))
            if diff > max_diff:
                max_diff = diff
    return max_diff

# Checks the selected action in every cell
def policies_equal(pi, piBase, airport_map):
    for x in range(airport_map.width()):
        for y in range(airport_map.height()):
            if pi.action(x, y) != piBase.action(x, y):
                return False
    return True

def helper(airport_map, theta, max_eval_steps):
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(0.8) #p=0.8

    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)

    # Values you can change:
    policy_solver.set_theta(theta)
    policy_solver.set_max_policy_evaluation_steps_per_iteration(max_eval_steps)

    # Set up initial state
    policy_solver.initialize()

    # NOTE: Commented out all the GUI as not really needed

    # Bind the drawer with the solver
    # policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    # policy_solver.set_policy_drawer(policy_drawer)
    
    # value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    # policy_solver.set_value_function_drawer(value_function_drawer)

    # metrics 
    t_start = time.time()  
    # Compute the solution
    v, pi = policy_solver.solve_policy()
    runtime = time.time() - t_start
    
    # # Save screen shot; this is in the current directory
    # policy_drawer.save_screenshot("policy_iteration_results.jpg")
    
    # # Wait for a key press
    # value_function_drawer.wait_for_key_press()

    return v, pi, runtime


# Q3e:
# Investigate different parameters
if __name__ == '__main__':
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()

    # Compute strict baseline solution
    # Baseline: theta = 0.0001, max_eval_steps = 200, runtime = 45.043092250823975
    theta_base = 1e-4
    max_steps_base = 200
    vBase, piBase, runtimeBase = helper(airport_map, theta_base, max_steps_base)
    print("Baseline: theta = {}, max_eval_steps = {}, runtime = {}".format(theta_base, max_steps_base, runtimeBase))

    # Grids to search over
    theta_grid = [1e-2, 1e-3, 1e-4, 1e-5]
    max_steps_grid = [10, 50, 100, 200]
    results = []

    for th in theta_grid: 
        for ms in max_steps_grid:
            v, pi, runtime = helper(airport_map, th, ms)

            # Compute abs diff
            max_abs_diff = max_abs_diff_vfunc(airport_map, v, vBase)

            # Policy match
            policy_same = policies_equal(pi, piBase, airport_map)

            results.append((th, ms, runtime, max_abs_diff, policy_same))

            print(f"theta={th:<5} max_steps={ms:<3} "
                  f"runtime={runtime:>6.3f}s  "
                  f"max|V-Vref|={max_abs_diff:>10.4f}  "
                  f"policy_same={policy_same}")
    
    out_csv = "q3e_results.csv"
    with open(out_csv, "w") as f:
        f.write("theta,max_eval_steps,runtime_s,max_abs_diff_v,policy_same\n")
        for row in results:
            f.write(",".join(map(str, row)) + "\n")

    print(f"\nSaved results to {out_csv}")