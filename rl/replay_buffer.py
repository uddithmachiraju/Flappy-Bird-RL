import torch
import numpy as np

# Replay buffer implementation
class ReplayBuffer:
    """
    Replay Buffer for storing experience tuples and sampling batches for training.
    
    Attributes:
        state (np.ndarray): Stores states.
        action (np.ndarray): Stores actions taken.
        reward (np.ndarray): Stores rewards received.
        next_state (np.ndarray): Stores next states.
        terminated (np.ndarray): Stores whether the episode ended after the action.
        pointer (int): Points to the next position to store data in the buffer.
        size (int): Tracks the current size of the buffer.
        max_size (int): Maximum number of experiences the buffer can hold.
    """
    def __init__(self, state_dim, action_dim, max_size = int(1e5)):
        """
        Initializes the Replay Buffer with the specified maximum size and dimensions.
        
        Args:
            state_dim (tuple): Dimension of the state space.
            action_dim (tuple): Dimension of the action space.
            max_size (int): Maximum number of experiences to store.
        """
        # Initialize buffer arrays
        self.state = np.zeros((max_size, state_dim), dtype = np.float32)       # Buffer for states
        self.action = np.zeros((max_size, action_dim), dtype = np.int32)       # Buffer for actions
        self.reward = np.zeros((max_size, 1), dtype = np.float32)               # Buffer for rewards
        self.next_state = np.zeros((max_size, state_dim), dtype = np.float32)  # Buffer for next states
        self.terminated = np.zeros((max_size, 1), dtype = np.float32)           # Buffer for termination signals

        # Pointers and size tracking
        self.pointer = 0            # Points to the index for the next insertion
        self.size = 0               # Tracks the current size of the buffer
        self.max_size = max_size    # Maximum buffer size

    def update(self, state, action, reward, next_state, terminated):
        """
        Stores a new experience in the buffer, overwriting old experiences if full.
        
        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            terminated (bool): Whether the episode ended.
        """
        try:
            # Ensure states and next states have the correct shape
            state = np.array(state, dtype=np.float32).reshape(self.state.shape[1:])
            next_state = np.array(next_state, dtype=np.float32).reshape(self.next_state.shape[1:])

            # Store the experience
            self.state[self.pointer] = state
            self.action[self.pointer] = action
            self.reward[self.pointer] = reward
            self.next_state[self.pointer] = next_state
            self.terminated[self.pointer] = terminated

            # Update the pointer and size
            self.pointer = (self.pointer + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

        except ValueError as e:
            print(f"Error updating replay buffer: {e}")
            print(f"State shape: {state.shape}, Expected shape: {self.state.shape[1:]}")
            print(f"Next state shape: {next_state.shape}, Expected shape: {self.next_state.shape[1:]}")
            raise

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: A tuple of torch tensors (state, action, reward, next_state, terminated).
        """
        # Randomly sample indices for the batch
        random_index = np.random.randint(0, self.size, batch_size)

        # Return sampled experiences as PyTorch tensors
        return (
            torch.FloatTensor(self.state[random_index]),        # States batch
            torch.FloatTensor(self.action[random_index]),       # Actions batch
            torch.FloatTensor(self.reward[random_index]),       # Rewards batch
            torch.FloatTensor(self.next_state[random_index]),   # Next states batch
            torch.FloatTensor(self.terminated[random_index])    # Termination signals batch
        )
