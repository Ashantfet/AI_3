# algorithm/random_agent.py

import numpy as np

class RandomAgent:
    def act(self, obs):
        return np.random.randint(0, 2, size=3)
