#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from common.scenarios import corridor_scenario
from common.airport_map_drawer import AirportMapDrawer

from p1.low_level_actions import LowLevelActionType

from td.sarsa import SARSA
from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)
    
    # Specify array of learners, renderers and policies
    learners = [None] * 2
    v_renderers = [None] * 2
    p_renderers = [None] * 2 
    pi = [None] * 2
    
    pi[0] = env.initial_policy()
    pi[0].set_epsilon(1)
    learners[0] = SARSA(env)
    learners[0].set_alpha(0.1)
    learners[0].set_experience_replay_buffer_size(64)
    learners[0].set_number_of_episodes(32)
    learners[0].set_initial_policy(pi[0])
    v_renderers[0] = ValueFunctionDrawer(learners[0].value_function(), drawer_height)    
    p_renderers[0] = LowLevelPolicyDrawer(learners[0].policy(), drawer_height)

    for i in range(10000):
        print(i)
        learners[0].find_policy()
        v_renderers[0].update()
        p_renderers[0].update()
        pi[0].set_epsilon(1/math.sqrt(1+0.25*i))


    # Reset for Q-learning
    random.seed(10)
    np.random.seed(10)
    airport_map, drawer_height = corridor_scenario()
    env_q = LowLevelEnvironment(airport_map)   

    pi[1] = env_q.initial_policy()
    pi[1].set_epsilon(1)
    learners[1] = QLearner(env_q)
    learners[1].set_alpha(0.1)
    learners[1].set_experience_replay_buffer_size(64)
    learners[1].set_number_of_episodes(32)
    learners[1].set_initial_policy(pi[1])      
    v_renderers[1] = ValueFunctionDrawer(learners[1].value_function(), drawer_height)    
    p_renderers[1] = LowLevelPolicyDrawer(learners[1].policy(), drawer_height)

    for i in range(10000):
        print(i)
        learners[1].find_policy()
        v_renderers[1].update()
        p_renderers[1].update()
        pi[1].set_epsilon(1/math.sqrt(1+0.25*i))

    output_dir = 'Coursework_02/Code/figures/2h'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    airport_map_drawer.save_screenshot(os.path.join(output_dir, 'scenario_map.pdf'))

    # SARSA results
    v_renderers[0].save_screenshot(os.path.join(output_dir, 'value_function_sarsa.pdf'))
    p_renderers[0].save_screenshot(os.path.join(output_dir, 'policy_sarsa.pdf'))

    # Q-Learner results
    v_renderers[1].save_screenshot(os.path.join(output_dir, 'value_function_qlearner.pdf'))
    p_renderers[1].save_screenshot(os.path.join(output_dir, 'policy_qlearner.pdf'))


    # Plots 
    v_sarsa = learners[0].value_function()._values
    v_qlearn = learners[1].value_function()._values

    # Average value per column (distance from goal)
    mask_s = ~np.isnan(v_sarsa)
    mask_q = ~np.isnan(v_qlearn)

    avg_sarsa = []
    avg_qlearn = []
    columns = []
    for x in range(v_sarsa.shape[0]):
        vals_s = v_sarsa[x, :][mask_s[x, :]]
        vals_q = v_qlearn[x, :][mask_q[x, :]]
        if len(vals_s) > 0 and len(vals_q) > 0:
            avg_sarsa.append(np.mean(vals_s))
            avg_qlearn.append(np.mean(vals_q))
            columns.append(x)

    plt.figure(figsize=(10, 5))
    plt.plot(columns, avg_qlearn, 'o-', label='Q-learning')
    plt.plot(columns, avg_sarsa, 's-', label='SARSA')
    plt.xlabel('Column (left = far from goal, right = near goal)')
    plt.ylabel('Average State Value')
    plt.title('Average State Value per Column: Q-learning vs SARSA')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'avg_value_comparison.pdf'))
    plt.show()


    # Difference heatmap
    diff = np.full_like(v_sarsa, float('nan'))
    mask = mask_s & mask_q
    diff[mask] = v_qlearn[mask] - v_sarsa[mask]

    plt.figure(figsize=(12, 4))
    plt.imshow(diff.T[::-1], cmap='Reds', aspect='auto')
    plt.colorbar(label='Q-learning − SARSA')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('State Value Difference (Q-learning minus SARSA)')
    plt.savefig(os.path.join(output_dir,'value_difference_heatmap.pdf'))
    plt.show()


    # Policy agreement map
    pi_sarsa = learners[0].policy()
    pi_qlearn = learners[1].policy()
    env_map = env.map()

    agree = np.full((env_map.width(), env_map.height()), float('nan'))
    for x in range(env_map.width()):
        for y in range(env_map.height()):
            if env_map.cell(x, y).is_obstruction() or env_map.cell(x, y).is_terminal():
                continue
            a_sarsa = pi_sarsa.action(x, y)
            a_qlearn = pi_qlearn.action(x, y)
            agree[x, y] = 1 if a_sarsa == a_qlearn else 0

    plt.figure(figsize=(12, 4))
    plt.imshow(agree.T[::-1], cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Green = agree, Red = disagree')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Policy Agreement: Q-learning vs SARSA')
    plt.savefig(os.path.join(output_dir,'policy_agreement.pdf'))
    plt.show()

    total = np.nansum(~np.isnan(agree))
    matching = np.nansum(agree == 1)
    print(f"Policy agreement: {matching}/{int(total)} cells ({100*matching/total:.1f}%)")


    # Action distribution comparison
    action_counts_sarsa = {}
    action_counts_qlearn = {}
    env_map = env.map()

    for x in range(env_map.width()):
        for y in range(env_map.height()):
            if env_map.cell(x, y).is_obstruction() or env_map.cell(x, y).is_terminal():
                continue
            a_s = pi_sarsa.action(x, y)
            a_q = pi_qlearn.action(x, y)
            action_counts_sarsa[a_s] = action_counts_sarsa.get(a_s, 0) + 1
            action_counts_qlearn[a_q] = action_counts_qlearn.get(a_q, 0) + 1

    actions = sorted(set(list(action_counts_sarsa.keys()) + list(action_counts_qlearn.keys())))
    labels = [LowLevelActionType(a).name for a in actions]
    sarsa_vals = [action_counts_sarsa.get(a, 0) for a in actions]
    qlearn_vals = [action_counts_qlearn.get(a, 0) for a in actions]

    x_pos = np.arange(len(actions))
    plt.figure(figsize=(10, 5))
    plt.bar(x_pos - 0.2, qlearn_vals, 0.4, label='Q-learning')
    plt.bar(x_pos + 0.2, sarsa_vals, 0.4, label='SARSA')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Number of cells')
    plt.title('Action Distribution: Q-learning vs SARSA')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'action_distribution.pdf'))
    plt.show()