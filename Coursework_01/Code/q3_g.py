#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

import time

from generalized_policy_iteration.value_iterator import ValueIterator
from common.scenarios import *
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

def calculate_policy_differences(env, pi_1, pi_2):
    map = env.map()
    diff = 0
    total = 0
    for x in range(map.width()):
        for y in range(map.height()):
            if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                continue
            total += 1
            if pi_1.action(x, y) != pi_2.action(x, y):
                diff += 1
    return diff, total


if __name__ == '__main__':
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(0.8)
    
    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)
    
    # Set up initial state
    policy_solver.initialize()

    t_policy_start = time.time() 
    # Compute the solution
    v_policy, pi_policy = policy_solver.solve_policy()
    t_policy_end = time.time()
    pi_runtime = t_policy_end - t_policy_start
    
    # Q3i: Add code to evaluate value iteration down here.
    vi_solver = ValueIterator(airport_environment)
    vi_solver.initialize()
    t_vi_start = time.time() 
    v_value_iteration, pi_value_iteration = vi_solver.solve_policy()
    t_vi_end = time.time()
    vi_runtime = t_vi_end - t_vi_start

    diff, total = calculate_policy_differences(airport_environment, pi_policy, pi_value_iteration)

    print(f'Policy iteration runtime: {pi_runtime:.2f} seconds')
    print(f'Value iteration runtime: {vi_runtime:.2f} seconds')
    print(f'Policy differences: {diff} out of {total} states')

    print("Policy iteration outer iterations:", policy_solver.policy_improvement_iterations())
    print("Policy iteration total evaluation sweeps:", policy_solver.total_policy_evaluation_sweeps())
    print("Value iteration sweeps:", vi_solver.value_iteration_sweeps())

    print("Policy iteration bellman updates:", policy_solver.bellman_updates())
    print("Value iteration bellman updates:", vi_solver.bellman_updates())