import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Define the environment
env = gym.make("CartPole-v0")

# Collect the demonstration data
demonstration_data = []
state = env.reset()
done = False
num_episodes = 10000
epsilon = 0.2
while not done:
    action = env.action_space.sample() # randomly choose an action
    # print(env.step(action))
    next_state, reward, done,_ = env.step(action)

    demonstration_data.append((state, action, reward))
    state = next_state

# Define the parameters of the DQN algorithm
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor

# Define the DQN network architecture
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DQN()

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=alpha)

# Initialize the replay buffer
replay_buffer = deque(maxlen=1000)

# Pre-train the DQN using the demonstration data
for state, action, reward in demonstration_data:
    state = torch.tensor(state, dtype=torch.float32)
    q_values = model(state)
    action = torch.tensor(action, dtype=torch.long)
    action = action.view(-1,1)
    q_values = q_values.gather(1, action)
    q_values = reward + gamma * torch.max(q_values)
    replay_buffer.append((state, q_values))


# Train the DQN using the replay buffer
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    while not done:
        # Choose an action using an epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(state)
            action = torch.argmax(q_values)
        
        # Perform the action and store the transition in the replay buffer
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        q_values = model(next_state)
        q_values[action] = reward + gamma * torch.max(q_values)
        replay_buffer.append((next_state, q_values))
