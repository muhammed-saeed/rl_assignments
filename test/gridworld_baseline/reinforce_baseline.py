import torch
import torch.nn as nn
import torch.optim as optim
from env import GridWorld

env = GridWorld(4, 4, (2, 0), 'east', [(3, 1)], [(0, 0), (0, 3), (1, 3)], [(2, 2), 'east', [(2, 1), (3, 1)]], ['m', 'l', 'r', 'f', 'pick', 'put'])
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

class BaselineNetwork(nn.Module):
    def __init__(self, state_size):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
state_size = 16
action_size = 6
num_episodes=1000
policy_network = PolicyNetwork(state_size, action_size)
baseline_network = BaselineNetwork(state_size)
optimizer = optim.Adam(policy_network.parameters())

actionString = ['m', 'l', 'r', 'f', 'pick', 'put']

for i in range(num_episodes):
    state = env.reset()
    done = False
    episode_rewards = []
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        #the state_shape is 4x4 need to be reshaped
        state_tensor = state_tensor.reshape(1,16)
        action_probs = policy_network(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        action_tensor = torch.tensor([action]) # convert action to tensor
        action_tensor = action_tensor.unsqueeze(0) # reshape tensor to (1,)
        # print(f"$$$$$$$$$$$$$ \n{action_tensor}")
        next_state, reward, done, _ = env.step(actionString[action_tensor])
        # print(actionString[action_tensor])
        episode_rewards.append(reward)

        # Update the policy network
        baseline = baseline_network(state_tensor).item()
        advantage = sum(episode_rewards) - baseline
        action_tensor = torch.tensor([action]) # convert action to tensor
        action_tensor = action_tensor.unsqueeze(1) # reshape tensor to (batch_size,1)
        policy_loss = -torch.log(action_probs[action]) * advantage
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Update the baseline network
        baseline_loss = (baseline - sum(episode_rewards)) ** 2
        optimizer.zero_grad()
        baseline_loss.backward()
        optimizer.step()

        state = next_state
