import numpy as np
from torch import nn
import torch

# Deep Q-Network model
class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the Deep Q-Network.
        Args:
            state_dim : Dimension of the state (flattened input size)
            action_dim : Dimension of the action
        """
        super(Network, self).__init__()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, action_dim)
        )

        # Initialize weights
        self.initialize_weights()

    def forward(self, input):
        """ Forward pass through the network. """
        # Pass input through fully connected layers
        output = self.fc_layers(input)
        return output

    def initialize_weights(self):
        """ Initialize weights using Xavier initialization for better convergence. """
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)