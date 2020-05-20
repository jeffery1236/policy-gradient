import numpy as np

class ReplayMemory():
    def __init__(self, mem_size, state_shape, num_actions):
        self.mem_size = mem_size
        self.mem_count = 0

        self.state_memory = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store(self, state, action, reward, new_state, done):
        index = self.mem_count % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.done_memory[index] = done

        self.mem_count += 1
    
    def sample(self, mini_batchsize):
        current_size = min(self.mem_count, self.mem_size)
        idx = np.random_choice(current_size, mini_batchsize, replace=False)

        states = self.state_memory[idx]
        actions = self.action_memory[idx]
        rewards = self.reward_memory[idx]
        new_states = self.new_state_memory[idx]
        dones = self.done_memory[idx]

        return states, actions, rewards, new_states, dones
    