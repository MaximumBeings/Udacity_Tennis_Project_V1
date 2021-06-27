# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24+2=28
        """
        class DDPGAgent:
            def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        """
        self.maddpg_agent = [DDPGAgent(24, 32, 16, 2, 26, 32, 16), 
                             DDPGAgent(24, 32, 16, 2, 26, 32, 16)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        
        import torch
        obs = []
        obs_full=[]
        action=[]
        reward=[]
        next_obs = []
        next_obs_full = []
        dones = []
        make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
        for x in samples:
            a,b,c,d,e,f,g = list(map(make_tensor, zip(*samples)))
            obs.append(a)
            obs_full.append(b)
            #print(obs_full)
            action.append(c)
            reward.append(d)
            next_obs.append(e)
            next_obs_full.append(f)
            dones.append(g)
        
        
            
            
        
        #obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)

        next_obs = torch.stack(next_obs)
        next_obs_full = torch.stack(next_obs_full)
        
        obs_full = torch.stack(obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs_full)
        #target_actions = torch.cat(target_actions, dim=1)
        
        #print("Target Actions:")
        #print(target_actions.shape)
        #print(target_actions)
        
        target_actions = torch.cat(target_actions, dim=1)
        
        #print("Target Actions Concatenated:")
        #print(target_actions.shape)
        #print(target_actions)
        #print()
        #print("next_obs:")
        #print(next_obs.shape)
        
        #a = next_obs.squeeze(1)
        #print(a)
        #print(a.shape)
        
        #target_critic_input = torch.cat((a  ,target_actions), dim=1).to(device)
        #print(target_critic_input)
        #print()
        next_obs_full= next_obs_full.squeeze(1)
        #print("Next Full Observations:")
        #print(next_obs_full.shape)
        #print(next_obs_full)
        
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=2).to(device)
        #print()
        #print("Target_Critic_Input:")
        #print(target_critic_input)
        #print(target_critic_input.shape)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        #print()
        #print(q_next)
        #print()
        #print("Reward:")
        reward = torch.cat(reward, dim=1)
        #print()
        #print(reward)
        #print()
        #print(reward.view(-1, 1))
        
        dones = torch.cat(dones, dim=1)
        #print()
        #print("Dones")
        #print(dones)
        y = reward.view(-1, 1) + self.discount_factor * q_next * (1 - dones.view(-1, 1))
        #print("Y:")
        #print(y)
        
        
        #print()
        action = torch.cat(action, dim=1)
        #print(action)
        #print()
        
        
        #print()
        obs_full= obs_full.squeeze(1)
        #print("Full Observations:")
        #print(obs_full.shape)
        #print(obs_full)
        
        critic_input = torch.cat((obs_full, action), dim=2).to(device)
        
        #print()
        #print("Critic_Input:")
        #print(critic_input)
        #print(critic_input.shape)
        
        #print("Q:")
        q = agent.critic(critic_input)
        #print(q)
        #print()
        #print(q.shape)
        
        #print()
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
        
        #print(q_input)
        #print()
        #print("Full Observations:")
        #print(obs_full.shape)
        #print(obs_full)
        #print("q_input:")
        #print(q_input)
        
        #print()
        
        q_input = torch.cat(q_input, dim=1)
        #q_input= q_input.unsqueeze(1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        #print()
        
        #print()
        q_input = q_input[0]
        #print(q_input)
        
        #print("q_input2:")
        #print()
        #q_input2 = torch.cat((obs_full, q_input), dim=0).to(device)
        #print()
        #print(q_input2)
        #obs_full= obs_full.squeeze()
        #print("Full Observations:")
        #print(obs_full[0][0].shape)
        #print(obs_full[0][0])
        #obs_full=obs_full[0]
        q_input2 = torch.cat((obs_full.view(576), q_input)).to(device)
        #C = torch.cat((A.view(1), B))
        #print()
        #print(q_input2)
        # get the policy gradient
        actor_loss = -agent.critic(q_input2[26:52]).mean()
        #print()
        #print(actor_loss)
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)
    def update_targets(self):
        #soft update targets
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)

        """
        target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        #soft update targets
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            


        """
        


