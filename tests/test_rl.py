import torch
import numpy as np
from rl.dqn import Network 
from rl.agent import Agent
from rl.replay_buffer import ReplayBuffer


def test_dqn_model():
    state_dim = (1, 128, 128)  # Example state dimensions (channels, height, width)
    action_dim = 2  # Example number of actions

    # Create the model
    model = Network(state_channels=state_dim[0], action_dim=action_dim)

    # Example input (batch_size=1)
    input_tensor = torch.randn(1, *state_dim)  # Batch of 1 image
    output = model(input_tensor)

    # Check the output shape (should match action_dim)
    assert output.shape == (1, action_dim), "Output shape is incorrect"
    print("DQN model test passed!")

def test_replay_buffer_and_agent():
    state_dim = (1, 128, 128)  # Example state dimensions (channels, height, width)
    action_dim = 2  # Example number of actions
    buffer_size = 100  # Buffer size

    # Create the replay buffer
    buffer = ReplayBuffer(state_dim, (action_dim,), max_size=buffer_size)

    # Create the model
    model = Network(state_channels=state_dim[0], action_dim=action_dim)

    # Create the agent
    agent = Agent(model=model, learning_rate=0.001, gamma=0.99)

    # Simulate adding experiences
    state = np.random.random(state_dim).astype(np.float32)
    action = np.array([1], dtype=np.int32)  # Taking action 1
    reward = np.array([1.0], dtype=np.float32)
    next_state = np.random.random(state_dim).astype(np.float32)
    terminated = np.array([0], dtype=np.float32)

    # Add experience to buffer
    buffer.update(state, action, reward, next_state, terminated)

    # Sample a batch of experiences
    states, actions, rewards, next_states, terminateds = buffer.sample(batch_size=1)

    # Check if the sampled batch has the correct shape
    assert states.shape == (1, *state_dim), "State batch shape is incorrect"
    assert actions.shape == (1, action_dim), "Action batch shape is incorrect"
    assert rewards.shape == (1, 1), "Reward batch shape is incorrect"
    assert next_states.shape == (1, *state_dim) , "Next state batch shape is incorrect"
    assert terminateds.shape == (1, 1), "Termination batch shape is incorrect"

    print("Replay buffer test passed!") 

    agent.train(states, actions, rewards, next_states, terminateds)

    # Check if training works (loss and optimizer should update the model)
    print("RL agent test passed!")

test_dqn_model()
test_replay_buffer_and_agent()
