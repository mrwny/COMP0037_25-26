#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''
import numpy as np
import random 
import matplotlib.pyplot as plt
from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from td.td_policy_predictor import TDPolicyPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)  
    
    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)  
    pe.evaluate()
    v_pe.update()  
    v_pe.update()  

    # Range of alpha values    
    alpha_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    
    num_values = len(alpha_values)
    
    td_predictors = [None] * num_values
    td_drawers = [None] * num_values
    
    # TD policy predictor
    for i in range (num_values):
        td_predictors[i] = TDPolicyPredictor(env)
        td_predictors[i].set_experience_replay_buffer_size(64)
        td_predictors[i].set_alpha(alpha_values[i])
        td_predictors[i].set_target_policy(pi)
        td_drawers[i] = ValueFunctionDrawer(td_predictors[i].value_function(), drawer_height)
        
    for e in range(400):
        for i in range(num_values):
            td_predictors[i].evaluate()
            td_drawers[i].update()        
 
    v_pe.save_screenshot("truth_pe.pdf")
    
    for i in range(num_values):
        td_drawers[i].update()
        td_drawers[i].save_screenshot(f"td-{alpha_values[i]:.2f}-pe.pdf")
    

    v_gt = pe.value_function()._values

    # Computing RMSE and MSE for each alpha
    rmses, mses = [], []
    for i in range(num_values):
        v_td = td_predictors[i].value_function()._values
        
        # Ignore NaN cells
        mask = ~np.isnan(v_gt) & ~np.isnan(v_td)
        differences = v_gt[mask] - v_td[mask]
        mse = np.mean(differences ** 2)
        rmse = np.sqrt(mse)
        mses.append(mse)
        rmses.append(rmse)
        print(f"alpha = {alpha_values[i]:.2f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(alpha_values, rmses, 'o-')
    ax1.set_xlabel(r'$\alpha$ (step size)')
    ax1.set_ylabel('RMSE')
    ax1.set_title(r'RMSE vs $\alpha$')
    ax1.grid(True)

    ax2.plot(alpha_values, mses, 's-', color='orange')
    ax2.set_xlabel(r'$\alpha$ (step size)')
    ax2.set_ylabel('MSE')
    ax2.set_title(r'MSE vs $\alpha$')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('error_vs_alpha.pdf')
    plt.show()
