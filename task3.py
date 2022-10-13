"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        # START EDITING HERE
        # You can add any other variables you need here

        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.mean   = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        thresh = 0.95
        t = np.sum(self.counts) 
        if t == 0:
            return np.random.randint(self.num_arms)
        else:
            for arm in range(self.num_arms):
                if self.counts[arm] != 0:
                    self.mean[arm] = self.values[arm]/self.counts[arm]
                    if self.mean[arm] >= thresh:
                        return arm
            return np.random.randint(self.num_arms)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.counts[arm_index] +=1
        if reward == 1:
            self.values[arm_index] += 1
            
        # END EDITING HERE
