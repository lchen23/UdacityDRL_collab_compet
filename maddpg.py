import numpy as np
import random
import copy
from collections import namedtuple, deque

from model_BN import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 5e-2  # for soft update of target parameters
LR_ACTOR = 1e-3  # actor learning rate
LR_CRITIC = 1e-3  # critic learning rate
WEIGHT_DECAY = 0  # L2 weight decay
UPDATE_EVERY = 4  # how often to update the network
UPDATE_TIMES = 4  # for each update, update the network how many times
PRETRAIN_MEM_SIZE = int(5e3) #Start learning only after 5000 experiences have been collected

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OU_Noise:
    """
    Ornstein-Uhlenbeck process
    """

    def __init__(self, size, seed, mu=0, sigma=0.2, dt=1, theta=0.15):
        """
        Initialize parameters and noise process
        :param size: length of random vector
        :param seed: random seed
        :param mu: mean value
        :param sigma: standard deviation
        :param dt: length of a timestep
        :param theta: inverse of a time decay constant
        """
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.dt = dt
        self.theta = theta
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        reset internal state to mean
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        update internal state and return it as a sample
        """
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state += dx
        return self.state


class Agent:
    """Multi-agent DDPG algo"""

    def __init__(self, num_agents, state_size, action_size, seed):
        """Initialize an Agent object.

                Params
                ======
                    num_agents (int): number of agents
                    state_size (int): dimension of each state
                    action_size (int): dimension of each action
                    seed (int): random seed
                """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ma_actions_dim = self.action_size * self.num_agents  # full action size of multi-agents
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # create multi-agents
        self.maddpg_agents = [SADDPG(num_agents, state_size, action_size, seed + i) for i in range(num_agents)]

    def reset(self):
        #reset each agent
        for agent in self.maddpg_agents:
            agent.reset()
        self.t_step = 0

    def act(self, states, add_noise=True):
        """
        Returns multi-agent actions for given state as per current policy.
        :param states: current multi-agent states
        :param add_noise: whether to add noise for exploration
        """
        actions = np.zeros((self.num_agents, self.action_size))
        for agent_idx, agent in enumerate(self.maddpg_agents):
            actions[agent_idx] = agent.act(states[agent_idx, :].reshape(1, -1), add_noise)

        return actions

    def step(self, states, actions, rewards, next_states, dones):
        # flatten multi-agents states
        ma_states = states.reshape(-1)
        ma_next_states = next_states.reshape(-1)

        # Save experience in replay memory
        self.memory.add(ma_states, states, actions, rewards, ma_next_states, next_states, dones)

        # update the network UPDATE_TIMES times for every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > PRETRAIN_MEM_SIZE:
                for i in range(UPDATE_TIMES):
                    for agent_no in range(self.num_agents):
                        experiences = self.memory.sample()
                        self.learn(experiences, agent_no, GAMMA)

    def learn(self, experiences, agent_no, gamma):
        # for learning MADDPG
        ma_states, states, actions, rewards, ma_next_states, next_states, dones = experiences

        ma_actions_next = torch.zeros_like(actions)  # init next action tensor
        for agent_idx, agent in enumerate(self.maddpg_agents):
            # get next action for each agent
            next_state = next_states[:, agent_idx, :]
            ma_actions_next[:, agent_idx, :] = agent.actor_target(next_state)

        # get predicted action for current agent
        agent = self.maddpg_agents[agent_no]
        agent_state = states[:, agent_no, :]
        ma_actions_pred = torch.zeros_like(actions)  # init predicted action tensor
        ma_actions_pred[:, agent_no, :] = agent.actor_local(agent_state)

        # flatten all action tensors
        ma_actions_pred = ma_actions_pred.view(-1, self.ma_actions_dim)
        ma_actions_next = ma_actions_next.view(-1, self.ma_actions_dim)
        ma_actions = actions.view(-1, self.ma_actions_dim)

        # get reward and done for each agent
        sa_reward = rewards[:, agent_no].view(-1, 1)
        sa_done = dones[:, agent_no].view(-1, 1)

        sa_experiences = (ma_states, ma_actions, ma_actions_pred, sa_reward, sa_done,
                          ma_next_states, ma_actions_next)
        
        # learn for single agent
        agent.learn(sa_experiences, gamma)

    def save_model(self, actor_name='checkpoint_actor_local_', critic_name='checkpoint_critic_local_'):
        for agent_idx, agent in enumerate(self.maddpg_agents):
            torch.save(agent.actor_local.state_dict(), actor_name + str(agent_idx) + '.pth')
            torch.save(agent.critic_local.state_dict(), critic_name + str(agent_idx) + '.pth')

    def load_model(self, actor_name='checkpoint_actor_local_', critic_name='checkpoint_critic_local_'):
        for agent_idx, agent in enumerate(self.maddpg_agents):
            if torch.cuda.is_available():
                # the model is trained on gpu, load all gpu tensors to gpu:
                agent.actor_local.load_state_dict(torch.load(actor_name + str(agent_idx) + '.pth'))
                agent.critic_local.load_state_dict(torch.load(critic_name + str(agent_idx) + '.pth'))
            else:
                # the model is trained on gpu, load all gpu tensors to cpu:
                agent.actor_local.load_state_dict(torch.load(actor_name + str(agent_idx) + '.pth',
                                                             map_location=lambda storage, loc: storage))
                agent.critic_local.load_state_dict(torch.load(critic_name + str(agent_idx) + '.pth',
                                                              map_location=lambda storage, loc: storage))


class SADDPG:
    """Single agent DDPG"""

    def __init__(self, num_agents, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network with target net
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network with target net
        self.critic_local = Critic(state_size * num_agents, action_size * num_agents, seed).to(device)
        self.critic_target = Critic(state_size * num_agents, action_size * num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Ornstein-Uhlenbeck noise
        self.noise = OU_Noise(action_size, seed)
        
        # Make sure target is initialized with the same weight as the source
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        :param state: current state
        :param add_noise: whether to add Ornstein-Uhlenbeck noise
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # add OU_noise to action to explore
        if add_noise:
            action_values += self.noise.sample()

        return np.clip(action_values, -1, 1)

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.

        :param experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma (float): discount factor
        """
        ma_states, ma_actions, ma_actions_pred, sa_reward, sa_done, ma_next_states, ma_actions_next \
            = experiences

        # ------------------- update critic ------------------- #
        # get predicted next state, actions and Q values from target network
        # actions_next = self.actor_target(next_states)
        Qtarget_next = self.critic_target(ma_next_states, ma_actions_next)
        # Compute Q targets for current states 
        Qtarget = sa_reward + (gamma * Qtarget_next * (1 - sa_done))

        # Get expected Q values from local model
        Qexpected = self.critic_local(ma_states, ma_actions)
        # calculate the batch loss
        critic_loss = F.mse_loss(Qexpected, Qtarget)

        # minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # backward pass
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # gradient clipping
        self.critic_optimizer.step()  # perform a single optimization step (parameter update)

        # ------------------- update actor ------------------- #
        # compute actor loss
        # actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(ma_states, ma_actions_pred).mean()

        # minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # backward pass
        self.actor_optimizer.step()  # perform a single optimization step (parameter update)

        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model (PyTorch model): weights will be copied from
        :param target_model (PyTorch model): weights will be copied to
        :param tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.

        :param buffer_size (int): maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["ma_states", "states", "actions", "rewards",
                                                  "ma_next_states", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, ma_states, states, actions, rewards, ma_next_states, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(ma_states, states, actions, rewards, ma_next_states, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.states for e in experiences if e is not None])).float().to(device)
        ma_states = torch.from_numpy(np.array([e.ma_states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_states for e in experiences if e is not None])).float().to(
            device)
        ma_next_states = torch.from_numpy(
            np.array([e.ma_next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.array([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (ma_states, states, actions, rewards, ma_next_states, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
