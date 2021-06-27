import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim =  1./np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size=24, action_size=2, seed=2, hidden_1 =200, hidden_2 = 200 ):
        """
        Parameters
        ----------
        state_size : TYPE, optional
            DESCRIPTION. The default is 24. - Action State Space
        action_size : TYPE, optional
            DESCRIPTION. The default is 2. - Number of Actions
        seed : TYPE, optional
            DESCRIPTION. The default is 2. - Random Seed
        hidden_1 : TYPE, optional
            DESCRIPTION. The default is 300 - Hidden Layer 1
        hidden_2 : TYPE, optional
            DESCRIPTION. The default is 200 - Hidden Layer 2

        Returns
        -------
        None.

        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.hidden_1 = nn.Linear(state_size, hidden_1)
        self.batchNormLayer_1 = nn.BatchNorm1d(hidden_1)
        
        self.hidden_2 = nn.Linear(hidden_1, hidden_2)
        self.batchNormLayer_2 = nn.BatchNorm1d(hidden_2)
        
        self.output_layer = nn.Linear(hidden_2, action_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Resets the parameters by setting a noise from distribution 
        following from its respective hidden unit size.
        Format for (-fx,fx) followed from the original paper.
        
        Returns
        -------
        None.

        """
        self.hidden_1.weight.data.uniform_(*hidden_init(self.hidden_1))
        self.hidden_2.weight.data.uniform_(*hidden_init(self.hidden_2))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        
    
    def forward(self, state):
        """
        

        Parameters
        ----------
        state : TYPE
            DESCRIPTION: State

        Returns
        -------
        None.

        """
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = state
       
        x = self.hidden_1(x)
        x = self.batchNormLayer_1(x)
        x = F.relu(x)
        
     
        x = self.hidden_2(x)
        x = self.batchNormLayer_2(x)
        x = F.relu(x)

        x = self.output_layer(x)
        res = torch.tanh(x)
        return res
        
        
class Critic(nn.Module):
    def __init__(self, state_size=24, action_size=2, seed=2, hidden_1 =200, hidden_2 = 200 ):
        """
        Parameters
        ----------
        state_size : TYPE, optional
            DESCRIPTION. The default is 24. - Action State Space
        action_size : TYPE, optional
            DESCRIPTION. The default is 2. - Number of Actions
        seed : TYPE, optional
            DESCRIPTION. The default is 2. - Random Seed
        hidden_1 : TYPE, optional
            DESCRIPTION. The default is 300 - Hidden Layer 1
        hidden_2 : TYPE, optional
            DESCRIPTION. The default is 200 - Hidden Layer 2

        Returns
        -------
        None.

        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.hidden_1 = nn.Linear(state_size, hidden_1)
        self.batchNormLayer_1 = nn.BatchNorm1d(hidden_1)
        
        self.hidden_2 = nn.Linear(hidden_1+ action_size, hidden_2)
        self.batchNormLayer_2 = nn.BatchNorm1d(hidden_2)
        
        self.output_layer = nn.Linear(hidden_2, action_size)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Resets the parameters by setting a noise from distribution 
        following from its respective hidden unit size.
        Format for (-fx,fx) followed from the original paper.
        
        Returns
        -------
        None.

        """
        self.hidden_1.weight.data.uniform_(*hidden_init(self.hidden_1))
        self.hidden_2.weight.data.uniform_(*hidden_init(self.hidden_2))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """
        
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        
        x = self.hidden_1(state)
        x = F.relu(x)
        x = self.batchNormLayer_1(x)
        
        x = torch.cat((x, action), dim=1) 
        x = self.hidden_2(x) 
        x = F.relu(x)
        
        res = self.output_layer(x)
        return res
        
        
        
        
        
        
        
        
        
        