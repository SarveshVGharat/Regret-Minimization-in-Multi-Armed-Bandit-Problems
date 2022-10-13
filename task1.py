"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
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

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need

def KL(p,q):
    if p == q or q == 0 or q == 1:
        return 0
    else:
        KL = (p* math.log((p/q) + 1e-6)) + ((1-p) * math.log(((1-p)/(1-q)) + 1e-6))
        return KL

def BinarySearch(p, thresh):
    low = p
    high = 1
    mid = low + (high - low)/2
    while abs(high - low) >= 1e-3:
        q = mid
        if abs(KL(p, q)-thresh) <= 1e-6:
            break
        elif KL(p, q) < thresh:
            low = mid + 1e-3
        else :                  
            high = mid - 1e-3
        mid = low + (high - low)/2
    return mid 

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.bonus  = np.zeros(num_arms)
        self.mean   = np.zeros(num_arms)
        self.ucb    = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        t = np.sum(self.counts) 
        
        if int(t) < int(self.num_arms):
            return int(t)
        else:
            self.mean  = self.values/self.counts
            self.bonus = np.sqrt(2*math.log(t)/self.counts)
            self.ucb   = self.mean + self.bonus
            return np.argmax(self.ucb)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.counts[arm_index] +=1
        if reward == 1:
            self.values[arm_index] += 1

        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.mean   = np.zeros(num_arms)
        self.klucb  = np.zeros(num_arms)
    
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        t = int(np.sum(self.counts))
        c = 3
        if int(t) < int(2*self.num_arms):
            if int(t) < int(self.num_arms):
                return int(t)
            else:
                return int(self.num_arms) - int(t)

        else:
            self.mean = self.values/self.counts
            for arm in range(self.num_arms):
                thresh = (math.log(t) + c*math.log(math.log(t)))/self.counts[arm]
                self.klucb[arm] = BinarySearch(self.mean[arm], thresh)
            return np.argmax(self.klucb)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.counts[arm_index] +=1
        if reward == 1:
            self.values[arm_index] += 1
            
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.beta   = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
 
        for arm in range(self.num_arms):
            self.beta[arm] = np.random.beta(self.values[arm] + 1, self.counts[arm] - self.values[arm] + 1)
        return np.argmax(self.beta)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        self.counts[arm_index] +=1
        if reward == 1:
            self.values[arm_index] += 1

        # END EDITING HERE
