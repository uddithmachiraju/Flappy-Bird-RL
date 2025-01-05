import os
import numpy as np
import torch

from rl.agent import Agent
from rl.replay_buffer import ReplayBuffer
from rl.dqn import Network
from config.rl_config import PARAMETERS
from core.game import Game

def train_agent(config):
    """
    Train the agent using the Flappy Bird environment.

    Args:
        config (dict): Training configuration parameters.
    """
    # Unpack configuration
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    buffer_size = config["buffer_size"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    gamma = config["gamma"]
    epsilon_start = config["epsilon_start"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps"]
    save_path = config["save_path"]

    # Initialize the model, agent, and replay buffer
    model = Network(state_dim = state_dim, action_dim = action_dim)
    target_model = Network(state_dim = state_dim, action_dim = action_dim)
    agent = Agent(model = model, model_class = target_model, learning_rate = learning_rate, gamma = gamma)
    buffer = ReplayBuffer(state_dim, action_dim, max_size = buffer_size) 

    # Initialize the game environment
    game_env = Game()

    while True:
        state_old = game_env.get_state() 
        action = agent.get_action(state_old, epsilon_start)
        reward, terminated, = game_env.run(human = False, AI = True, action = action) 
        new_state = game_env.get_state()  
        buffer.update(state_old, action, reward, new_state, terminated) 

        # Sample a batch and train the agent
        if buffer.size > batch_size:
            states, actions, rewards, next_states, terminateds = buffer.sample(batch_size)
            agent.train(states, actions, rewards, next_states, terminateds)

        # Decay epsilon
        epsilon_start = max(epsilon_min, epsilon_start * epsilon_decay)

        print(f"Reward: {reward:.2f}, Epsilon: {epsilon_start:.4f}, Action: {action}") 

        if terminated == False: game_env.reset()

    # Save the trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_agent(PARAMETERS)
