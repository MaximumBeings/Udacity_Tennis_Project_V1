import numpy as np
from utils2 import OUNoise
from model2 import Actor, Critic
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 



TAU = 1e-2
LR_CRITIC = 2e-4
LR_ACTOR = 2e-4
WEIGHT_DECAY = 0.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed):
        """
        Main DDPG Agent

        Parameters
        ----------
        state_size : TYPE
            DESCRIPTION: State Observation Space
        action_size : TYPE
            DESCRIPTION: Number of Actions
        random_seed : TYPE
            DESCRIPTION: Random Seed

        Returns
        -------
        None.

        """
        self.state = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)
        
        self.noise = OUNoise(action_size, random_seed)
        
    def reset(self):
        """
        Resets Noise 

        Returns
        -------
        None.

        """
        self.noise.reset()
        
    def act(self, state, add_noise=True):
        """
        Returns Actions Given State        
        
        Parameters
        ----------
        state : TYPE
            DESCRIPTION: State Observation Space
        add_noise : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if(add_noise):
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def learn(self, experiences, gamma):
        """
        Learn from Experience

        Parameters
        ----------
        experiences : TYPE
            DESCRIPTION: NamedTuple set of experiences (S A R S' DONE FLAG)
        gamma : TYPE
            DESCRIPTION: Gamma

        Returns
        -------
        None.

        """
        #Unpack experience tuple
        states, actions, rewards, next_states, dones = experiences
        
        #update critic network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        #Compute and minimize critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        actions_pred = self.actor_local(states) #
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        #Update Target
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #Update Target Networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU) 
        
        
        
    def soft_update(self, local_model, target_model, tau):
        """
        Parameters
        ----------
        local_model : TYPE
            DESCRIPTION: Source model to copy from
        target_model : TYPE
            DESCRIPTION: Source model to copy to
        tau : TYPE
            DESCRIPTION: Interpolationparameters

        Returns
        -------
        None.

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_copy_weights(self, target, source):
        """
        Parameters
        ----------
        local_model : TYPE
            DESCRIPTION: Source model to copy weights from
        target_model : TYPE
            DESCRIPTION: Source model to copy weights
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
