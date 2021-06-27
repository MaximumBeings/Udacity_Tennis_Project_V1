import torch
from agent2 import Agent
from utils2 import ReplayBuffer


GAMMA = 0.99
BUFFER_SIZE = 1000000
BATCH_SIZE = 200


class MADDPG():
    def __init__(self, num_agents=2, state_size=24, action_size=2, random_seed=2):
        """
        Parameters
        ----------
        num_agents : TYPE, optional
            DESCRIPTION. The default is 2.
        state_size : TYPE, optional
            DESCRIPTION. The default is 24.
        action_size : TYPE, optional
            DESCRIPTION. The default is 2.
        random_seed : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        self.num_agents = num_agents
        self.agents = [Agent(state_size, action_size, random_seed) for x in range(self.num_agents)]
        self.memory = ReplayBuffer(action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=random_seed)
        
        
    def act(self, states, add_noise=True):
        """
        Implement Actions on all the Agents in the Enviroment

        Parameters
        ----------
        states : TYPE
            DESCRIPTION: Observed State of the Agent
        add_noise : TYPE, optional
            DESCRIPTION: The default is True - OUNoise

        Returns
        -------
        None.

        """
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state)
            actions.append(action)
        return actions
        
    def reset(self):
        """
        Reset the noise level for all the agents

        Returns
        -------
        None.

        """
        
        for x in self.agents:
            x.reset()
            
    def step(self, states, actions, rewards, next_states, dones):
        """
        Saves Experience in ReplayBuffer

        Parameters
        ----------
        states : TYPE
            DESCRIPTION: Current State
        actions : TYPE
            DESCRIPTION: Actions Taken by Agents from Current State
        rewards : TYPE
            DESCRIPTION: Rewards Earned from Taking Action in Current State
        next_states : TYPE
            DESCRIPTION: Next States
        dones : TYPE
            DESCRIPTION: Whether Episode Terminated or Not

        Returns
        -------
        None.

        """
        
        for x in range(self.num_agents):
            self.memory.add(states[x], actions[x], rewards[x], next_states[x], dones[x])
        
        if(len(self.memory) > BATCH_SIZE):
            for x in range(self.num_agents):
                experience = self.memory.sample()
                self.learn(experience)
                
    def learn(self, experiences, gamma=GAMMA):
        """
        Function to Learn from Agent Experiences

        Parameters
        ----------
        experiences : TYPE
            DESCRIPTION.
        gamma : TYPE, optional
            DESCRIPTION. The default is GAMMA.

        Returns
        -------
        None.

        """
        for agent in self.agents:
            agent.learn(experiences,gamma)
        
     
    def saveCheckPoints(self, isDone):
        """
        Save the checkPoint weights of the Agents at intervals

        Parameters
        ----------
        isDone : TYPE
            DESCRIPTION: Boolean Flag indicating whether to save to checkPoint or not.

        Returns
        -------
        None.

        """
        if(isDone == False):
            for x, agent in enumerate(self.agents):
                torch.save(agent.actor_local.state_dict(), f"../models/checkpoint/actor_agent_{x}.pth")
                torch.save(agent.critic_local.state_dict(), f"../models/checkpoint/critic_agent_{x}.pth")
        else:
            for x, agent in enumerate(self.agents):
                torch.save(agent.actor_local.state_dict(), f"../models/final/actor_agent_{x}.pth")
                torch.save(agent.critic_local.state_dict(), f"../models/final/critic_agent_{x}.pth")
    
    
    def loadCheckPoints(self, isFinal=False):
        """
        Loads Checkpoints for agents

        Parameters
        ----------
        isFinal : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        if(isFinal):
            for x, agent in enumerate(self.agents):
                agent.actor_local.load_state_dict(torch.load(f"../models/final/actor_agent_{x}.pth"))
                agent.critic_local.load_state_dict(torch.load(f"../models/final/critic_agent_{x}.pth"))
        else:
            for x, agent in enumerate(self.agents):
                agent.actor_local.load_state_dict(torch.load(f"../models/checkpoint/actor_agent_{x}.pth"))
                agent.critic_local.load_state_dict(torch.load(f"../models/checkpoint/critic_agent_{x}.pth"))
                

        
        
        
        
        
        