import numpy as np
import copy
import random
from collections import deque, namedtuple

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """
    Orsten-Uhlenbeck Process.
    """
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.1):
        """
        Initialize Parameters and Noise Process.

        Parameters
        ----------
        size : TYPE
            DESCRIPTION.
        seed : TYPE
            DESCRIPTION.
        mu : TYPE, optional
            DESCRIPTION. The default is 0.0.
        theta : TYPE, optional
            DESCRIPTION. The default is 0.15.
        sigma : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        """
        self.mu = mu * np.ones(size)
        self.theta = theta 
        self.sigma = sigma 
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        """
        Reset the internal state (=noise) to mean (mu).

        Returns
        -------
        None.

        """
        self.state  = copy.copy(self.mu)
        
    def sample(self):
        """
        Update internal state and return it as a noise sample. 

        Returns
        -------
        None.

        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        

class ReplayBuffer():
    """
    ReplayBuffer
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Parameters
        ----------
        action_size : TYPE
            DESCRIPTION: Number of actions
        buffer_size : TYPE
            DESCRIPTION: Maximum Length of Buffer
        batch_size : TYPE
            DESCRIPTION: MinBatch Size
        seed : TYPE
            DESCRIPTION: Random Seed

        Returns
        -------
        None.
        """
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.replay_memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state","done"])

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to existing memory
        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        action : TYPE
            DESCRIPTION.
        reward : TYPE
            DESCRIPTION.
        next_state : TYPE
            DESCRIPTION.
        done : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        trajectory = self.experience(state, action, reward, next_state, done)
        self.replay_memory.append(trajectory)
    
    def sample(self):
        """Randomly picks minibatches within the replay_buffer of size mini_batch"""
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):#override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)
    
    
        
        
        


















