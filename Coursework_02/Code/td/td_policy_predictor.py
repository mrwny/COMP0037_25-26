'''
Created on 8 Mar 2023

@author: ucacsjj
'''

from monte_carlo.episode_sampler import EpisodeSampler

from .td_algorithm_base import TDAlgorithmBase

class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment):
        
        TDAlgorithmBase.__init__(self, environment)
        
        self._minibatch_buffer= [None]
                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            
    def _update_value_function_from_episode(self, episode):

        alpha = self._alpha
        gamma = self._gamma

        for t in range(episode.number_of_steps()):

            s = episode.state(t)
            r = episode.reward(t)
            coords = s.coords()
            v = self._v.value(coords[0], coords[1])

            if (t < episode.number_of_steps() - 1) and (episode.state(t + 1) is not None):
                s_prime = episode.state(t + 1)
                coords_prime = s_prime.coords()
                v_prime = self._v.value(coords_prime[0], coords_prime[1])
            else:
                v_prime = 0

            delta = r + gamma * v_prime - v
            new_v = v + alpha * delta
            self._v.set_value(coords[0], coords[1], new_v)



