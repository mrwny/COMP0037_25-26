'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from p2.low_level_actions import LowLevelActionType
from .dynamic_programming_base import DynamicProgrammingBase

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
        self._value_iteration_sweeps = 0
        self._bellman_updates = 0
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):
        self._value_iteration_sweeps = 0
        self._bellman_updates = 0

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3f:
    # Finish the implementation of the methods below.
    
    def _compute_optimal_value_function(self):

        # This method returns no value.
        # The method updates self._pi

                
        # Get the environment and map
        environment = self._environment
        map = environment.map()
        
        # Execute the loop at least once
        
        iteration = 0
        
        while True:
            
            delta = 0

            # Sweep systematically over all the states            
            for x in range(map.width()):
                for y in range(map.height()):
                    
                    # We skip obstructions and terminals. If a cell is obstructed,
                    # there's no action the robot can take to access it, so it doesn't
                    # count. If the cell is terminal, it executes the terminal action
                    # state. The value of the value of the terminal cell is the reward.
                    # The reward itself was set up as part of the initial conditions for the
                    # value function.
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue
                                       
                    # Unfortunately the need to use coordinates is a bit inefficient, due
                    # to legacy code
                    cell = (x, y)
                    
                    # Get the previous value function
                    old_v = self._v.value(x, y)
                    max_v = -float('inf')
                 
                    #find max
                    # for action in range(8):

                    for action in LowLevelActionType:
                        if action > LowLevelActionType.TERMINATE-1:
                            continue
                        # Compute p(s',r|s,a)
                        s_prime, r, p = environment.next_state_and_reward_distribution(cell, action)

                        # Sum over the rewards
                        new_v = 0
                        for t in range(len(p)):
                            sc = s_prime[t].coords()
                            new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1])) 

                        self._bellman_updates += 1
                        max_v = max(max_v, new_v)
                  
                            
                    # Set the new value in the value function
                    self._v.set_value(x, y, max_v)
                                        
                    # Update the maximum deviation
                    delta = max(delta, abs(old_v-max_v))
 
            # Increment the policy evaluation counter        
            iteration += 1
            
            self._value_iteration_sweeps = iteration
            print(f'Finished value iteration sweep {iteration}')
            
            # Terminate the loop if the change was very small
            if delta < self._theta:
                break
                
            # Terminate the loop if the maximum number of iterations is met. Generate
            # a warning
            if iteration >= self._max_optimal_value_function_iterations:
                print('Maximum number of iterations exceeded')
                break
    


    def _extract_policy(self):

        # This method returns no value.
        # The policy is in self._pi

        environment = self._environment
        map = environment.map()

        for x in range(map.width()):
            for y in range(map.height()):

                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue

                cell = (x, y)
                best_q = -float('inf')
                best_action = None

                for action in LowLevelActionType:
                    if action > LowLevelActionType.TERMINATE-1:
                        continue
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, action)
                    
                    q = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        q += p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))
                    
                    if q > best_q:
                        best_q, best_action = q, action

                self._pi.set_action(x, y, best_action)


    def value_iteration_sweeps(self):
        return self._value_iteration_sweeps
    
    def bellman_updates(self):
        return self._bellman_updates