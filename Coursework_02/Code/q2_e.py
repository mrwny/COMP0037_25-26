#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math
import os
import random
import matplotlib.pyplot as plt

import numpy as np

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


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

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()
    
    # Select the controller
    policy_learner = QLearner(env)   
    policy_learner.set_initial_policy(pi)

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)
    
    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)
    total_times = []
    for i in range(40):
        print(i)
        times = policy_learner.find_policy()
        total_times.extend(times)
        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        pi.set_epsilon(1/math.sqrt(1+0.25*i))
        print(f"epsilon={1/math.sqrt(1+i)};alpha={policy_learner.alpha()}")

    # latex formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })

    # Convert to numpy and milliseconds for readability
    times = np.array(total_times, dtype=float)
    times_ms = 1000.0 * times
    episodes = np.arange(1, len(times_ms) + 1)

    if len(times_ms) == 0:
        print("No timings recorded.")
    else:
        os.makedirs("figures/2e", exist_ok=True)

        plt.figure(figsize=(10, 4))
    
        plt.scatter(episodes, times_ms, s=10, alpha=0.7, label="Episode duration")
        
        plt.xlabel("Episode index")
        plt.ylabel("Time (ms)")
        plt.title("Q-learning episode timing")
        plt.grid(True, alpha=0.3)

        window = 25
        if len(times_ms) >= window:
            ma = np.convolve(times_ms, np.ones(window) / window, mode="valid")
            ma_x = np.arange(window, len(times_ms) + 1)
            plt.plot(ma_x, ma, lw=2.0, color='red', label=f"{window}-episode moving average")

        plt.legend()
        plt.tight_layout()
        
        plt.savefig("figures/2e/episode_times_scatter.pdf", format='pdf', bbox_inches='tight')

        plt.show()
        
        if len(times_ms) >= 100:
            avg_100 = np.mean(times_ms[max(0, 100-10):100+10])
            print(f"Average time around episode 100 (iterations ~3): {avg_100:.2f} ms")
            
        if len(times_ms) >= 200:
            avg_200 = np.mean(times_ms[190:210])
            print(f"Average time around episode 200 (iterations ~6): {avg_200:.2f} ms")
            
        if len(times_ms) >= 400:
            avg_400 = np.mean(times_ms[390:410])
            print(f"Average time around episode 400 (iterations ~12): {avg_400:.2f} ms")
            
        # Compute the "stabilized" final average over the last 20% of episodes
        if len(times_ms) > 100:
            idx_80_percent = int(len(times_ms) * 0.8)
            final_avg = np.mean(times_ms[idx_80_percent:])
            final_std = np.std(times_ms[idx_80_percent:])
            print(f"Stabilized final average (last 20%): {final_avg:.2f} ms")
            print(f"Final stabilized variance (std dev): {final_std:.2f} ms")