import gym
import gym_gridworlds
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Collect expert demonstrations
env = gym.make('CartPole-v0')
# env  = gym.make('Gridworld-v0')
expert_data = []
for _ in range(100):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() # expert policy
        next_obs, reward, done, _ = env.step(action)
        expert_data.append((obs, action))
        obs = next_obs

# Split the data into training and testing sets
np.random.shuffle(expert_data)
split = int(0.8 * len(expert_data))
train_data = expert_data[:split]
test_data = expert_data[split:]

# Define the training loop
# policy = Policy(4, 32, 2)
policy = Policy(4, 32, 4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters())

for epoch in range(1):
    running_loss = 0.0
    for obs, action in train_data:
        optimizer.zero_grad()
        action_pred = policy(torch.Tensor(obs))
        loss = criterion(action_pred, torch.argmax(torch.Tensor([action])))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1} Loss: {running_loss / len(train_data)}')

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for obs, action in test_data:
        action_pred = policy(torch.Tensor(obs))
        # print(action_pred)
        _, pred = torch.max(action_pred, 0)
        total += 1
        correct += (pred == action).item()
    print(f'Test Accuracy: {correct / total}')

# REINFORCE with Baseline Algorithm
# REINFORCE with Baseline Algorithm
baseline = 0
policy.train()
for episode in range(10000):
    obs = env.reset()
    done = False
    rewards = []
    log_probs = []
    while not done:
        action_pred = policy(torch.Tensor(obs))
        prob = nn.functional.softmax(action_pred, dim=-1)
        action = torch.multinomial(prob, 1).squeeze()
        # print(action)
        # print(prob)
        log_prob = torch.log(prob[action.item()])




        next_obs, reward, done, _ = env.step(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)
        obs = next_obs

    # Compute the returns
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + 0.99 * R
        # print(R)
        returns.insert(0, R)
    returns = torch.Tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Update the policy
    optimizer.zero_grad()
    loss = 0
    for log_prob, R in zip(log_probs, returns):
        advantage = R - baseline
        loss += -log_prob * advantage
    loss /= len(log_probs)
    loss.backward()
    optimizer.step()

    # Update the baseline
    baseline = 0.99 * baseline + 0.01 * returns.mean()


# Evaluate the agent
policy.eval()
obs = env.reset()
done = False
total_reward = 0
while not done:
    env.render() # added this line to visualize the agent
    action_pred = policy(torch.Tensor(obs))
    _, action = torch.max(action_pred, 0)
    obs, reward, done, _ = env.step(action.item())
    total_reward += reward

print(f'Total Reward: {total_reward}')
