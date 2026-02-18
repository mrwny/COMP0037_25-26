'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from collections import deque
from math import sqrt
from queue import PriorityQueue

from .occupancy_grid import OccupancyGrid
from .planner_base import PlannerBase

class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()  # type: ignore

    # Q1d:
    # Modify this class to finish implementing Dijkstra
    def push_cell_onto_queue(self, cell):
        # Insert the cell into the priority queue, keyed by its cost-to-come (path_cost).
        # PriorityQueue sorts by the first element of the tuple, so cells with
        # lower path_cost will be retrieved first by Q.GetFirst().
        self.priority_queue.put((cell.path_cost, cell))

    def is_queue_empty(self):
        # Check whether the priority queue has any cells remaining.
        return self.priority_queue.empty()

    def pop_cell_from_queue(self):
        # Retrieve and remove the cell with the lowest cost-to-come.
        # This corresponds to Q.GetFirst() in the pseudocode.
        cell = self.priority_queue.get()[1]
        return cell

    def resolve_duplicate(self, cell, parent_cell):
        # This implements the shortest path rewriting step (line 13).
        # When a cell that has already been visited is encountered again,
        # we check if reaching it via parent_cell offers a lower cost-to-come.
        # If so, we update the cell's parent and path_cost.

        # Compute the cost-to-come if we were to reach cell via parent_cell
        l_cost = self.compute_l_stage_additive_cost(parent_cell, cell)
        new_cost = parent_cell.path_cost + l_cost

        # If the new path is cheaper, rewrite the parent and cost
        if new_cost < cell.path_cost:
            cell.set_parent(parent_cell)
            cell.path_cost = new_cost
            # Re-insert the cell with the updated cost so it can be
            # re-expanded at the correct priority
            self.push_cell_onto_queue(cell)

    def mark_cell_as_visited_and_record_parent(self, cell, parent_cell):
        # Override the base class method to also compute and store the
        # cost-to-come when a cell is first visited.
        # This corresponds to the modification at line 10 of the pseudocode.

        # Call the base class to set the label and parent
        PlannerBase.mark_cell_as_visited_and_record_parent(self, cell, parent_cell)

        # Compute and store the cost-to-come: C(x') = C(x) + l(x, x')
        if parent_cell is not None:
            l_cost = self.compute_l_stage_additive_cost(parent_cell, cell)
            cell.path_cost = parent_cell.path_cost + l_cost
        else:
            cell.path_cost = 0
            