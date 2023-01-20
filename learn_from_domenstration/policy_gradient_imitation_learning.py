import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# Collect expert demonstrations
env = gym.make('CartPole-v0')
expert_data = []
for _ in range(1000):
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
policy = Policy(4, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters())
for epoch in range(100):
    running_loss = 0.0
    for obs, action in train_data:
        optimizer.zero_grad()
        action_pred = policy(torch.Tensor(obs))
        # print(action_pred)
        loss = criterion(action_pred, torch.argmax(torch.Tensor([action])))
        #we are using argmax since the target is a single action only
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
        _, pred = torch.max(action_pred, 1)
        total += 1
        correct += (pred == action).item()
    print(f'Test Accuracy: {correct / total}')
