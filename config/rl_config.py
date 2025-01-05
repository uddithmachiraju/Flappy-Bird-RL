import torch

# Hyperparameters for training
PARAMETERS = {
    "state_dim": 7, # (1, 128, 128),  # Input state dimensions (channels, height, width)
    "action_dim": 2,             # Number of possible actions
    "buffer_size": 1000,         # Maximum size of the replay buffer
    "batch_size": 1,            # Batch size for training
    "learning_rate": 0.0001,      # Learning rate for the optimizer
    "gamma": 0.9,               # Discount factor for future rewards
    "epsilon_start": 1.0,        # Starting value of epsilon for exploration
    "epsilon_min": 0.1,          # Minimum value of epsilon
    "epsilon_decay": 0.995,      # Decay rate for epsilon 
    "num_episodes": 1000,        # Total number of episodes
    "max_steps": 500,            # Max steps per episode
    "save_path": "logs/checkpoints/model_checkpoint.pth",  # Path to save the trained model
}