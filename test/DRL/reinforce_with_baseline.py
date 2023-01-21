import gym
import torch
import torch.nn as nn
import torch.optim as optim

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=-1)

# Define the value network (baseline)
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the policy and value networks
policy_network = PolicyNetwork(4, 2)
value_network = ValueNetwork(4)

# Define the optimizers
policy_optimizer = optim.Adam(policy_network.parameters())
value_optimizer = optim.Adam(value_network.parameters())

# Create the CartPole environment
env = gym.make('CartPole-v0')

# Hyperparameters
num_episodes = 1000
discount_factor = 0.99

# Train the agent
for episode in range(num_episodes):
    # Initialize the episode
    state = env.reset()
    episode_reward = 0
    episode_loss = 0

    while True:
        # Get the action probabilities from the policy network
        state_tensor = torch.FloatTensor(state)
        action_probs = policy_network(state_tensor)

        # Sample an action from the action probabilities
        action = torch.multinomial(action_probs, num_samples=1).item()

        # Take the action and get the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Compute the advantage
        value = value_network(state_tensor)
        # print(value)
        next_value = value_network(torch.FloatTensor(next_state)).item()
        advantage = reward + (1 - done) * discount_factor * next_value - value

        # Compute the loss for the policy network
        policy_loss = -torch.log(action_probs[action]) * advantage

        # Optimize the policy network
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph = True)
        policy_optimizer.step()

        # Compute the loss for the value network
        value_loss = (value - (reward + (1 - done) * discount_factor * next_value)) ** 2

        # Optimize the value network
        value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        value_optimizer.step()
