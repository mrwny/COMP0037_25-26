'''
Created on 2 Jan 2022

@author: ucacsjj
'''

import math
from typing import override

from .dijkstra_planner import DijkstraPlanner
from .occupancy_grid import OccupancyGrid

class AStarPlanner(DijkstraPlanner):
    def __init__(self, occupancy_grid: OccupancyGrid):
        DijkstraPlanner.__init__(self, occupancy_grid)

    # Q2d:
    # Complete implementation of A*.

    @override
    def push_cell_onto_queue(self, cell):
        # Insert the cell into the priority queue, keyed by its f-value (g + h).
        # The g-value is the path_cost (cost-to-come), and the h-value is the heuristic cost-to-go.
        f_value = cell.path_cost + self.compute_heuristic_cost_to_go(cell, self.goal)
        self.priority_queue.put((f_value, cell))
    
    @override
    def resolve_duplicate(self, cell, parent_cell):
        # Compute the g-value if we were to reach cell via parent_cell
        l_cost = self.compute_l_stage_additive_cost(parent_cell, cell)
        new_g_cost = parent_cell.path_cost + l_cost

        # If the new path is cheaper, rewrite the parent and cost
        if new_g_cost < cell.path_cost:
            cell.set_parent(parent_cell)
            cell.path_cost = new_g_cost
            # Re-insert the cell with the updated f-value so it can be
            # re-expanded at the correct priority
            self.push_cell_onto_queue(cell)
    
    def compute_heuristic_cost_to_go(self, cell, goal_cell):
        # Compute the heuristic cost-to-go (h-value) from the given cell to the goal.
        # Here we use the Euclidean distance as the heuristic.
        
        cell_coords = cell.coords()
        goal_coords = goal_cell.coords()
        
        h_cost = math.sqrt((cell_coords[0] - goal_coords[0]) ** 2 + (cell_coords[1] - goal_coords[1]) ** 2)
        return h_cost
