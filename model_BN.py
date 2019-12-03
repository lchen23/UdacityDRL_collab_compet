import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(layer):
    n_input = layer.weight.data.size()[0]
    weight_max = 1. / np.sqrt(n_input)
    return (-weight_max, weight_max)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # set up fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        # init weights for each layer
        self.fc1.weight.data.uniform_(*weights_init(self.fc1))
        self.fc2.weight.data.uniform_(*weights_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # add hidden layers into a sequential container
        self.net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            self.bn1,
            self.fc2,
            nn.ReLU(inplace=True),
            self.bn2,
            self.fc3
        )
    def forward(self, state):
        """Build an actor network that maps state -> action values."""
        x = self.net(state)
        x = torch.tanh(x)
        return x

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, ma_state_size, ma_action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # set up fully connected layers
        self.bn0 = nn.BatchNorm1d(ma_state_size)
        self.fc1 = nn.Linear(ma_state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + ma_action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        # init weights for each layer
        self.fc1.weight.data.uniform_(*weights_init(self.fc1))
        self.fc2.weight.data.uniform_(*weights_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic network that maps (state, action) pairs -> Q-values."""
        x = self.bn0(state)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)