import torch 
import numpy as np 
from torch import nn

class Agent:
    def __init__(self, model, model_class, learning_rate, gamma):
        """
        Args:
            model (nn.Module): The primary Q-network.
            model_class (type): The class of the model (e.g., Network).
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = model
        self.target_model = model_class  # Instantiate the same model class
        self.target_model.load_state_dict(self.model.state_dict())  # Copy weights
        self.target_model.eval()  # Set target model to evaluation mode

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def soft_update_target_network(self, tau=0.01):
        """
        Soft-update the target network: θ_target = τ * θ_policy + (1 - τ) * θ_target
        Args:
            tau (float): Soft-update rate (default=0.01).
        """
        for target_param, policy_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def train(self, states, actions, rewards, next_states, terminated):
        preds = self.model(states)
        target = preds.clone().detach()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]

        Q_new = rewards + self.gamma * max_next_q_values * (1 - terminated)
        action_indices = torch.argmax(actions, dim=1)
        target[range(states.size(0)), action_indices] = Q_new

        predicted_q_values = preds[range(states.size(0)), action_indices]
        loss = self.loss_func(predicted_q_values, Q_new)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.soft_update_target_network()

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1])  # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(self.model(state_tensor)).item()  # Exploit
