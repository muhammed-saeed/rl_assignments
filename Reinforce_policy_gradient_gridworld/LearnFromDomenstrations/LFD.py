

import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd

from read_env_json import read_env_sol_json
from memory import get_memory

import numpy as np

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


mode = "train"
train_path = "/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/task/task"
train_target_path = "/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/task/solution"
actions = ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']

# Define a function to get batches of data


def get_batch(batch_size, data):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# Train the network using batches


class Policy(nn.Module):
    def __init__(self, input_size=32, output_size=6):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine3 = nn.Linear(128, output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # x.requires_grads = True
        # x.requires_grad = True
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = self.dropout(x)
        x = F.relu(x)

        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)


state_size = 32
numEpochs = 100
batch_size = 25
policy = Policy(input_size=state_size, output_size=6)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
training_data = get_memory()

new_training_data = [row[:2] for row in training_data]
print(new_training_data[0])

losses = []


def get_batch(batch_size, data):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


batch = get_batch(batch_size, new_training_data)
print(batch)
print(new_training_data[0][0].flatten())

for epoch in range(5):
    loss_sum = 0
    i = 0
    for inputs, labelss in (training_data[:50]):
        i += 1
        # loss = 0
        inputs = torch.from_numpy(inputs).float()
        # print(labelss)
        labels = np.zeros(6)
        labels[int(labelss)] = 1
        # labels = one_hot(6, labelss)
        labels = torch.from_numpy(labels)

        # note [labels] since the output is a tensor
        optimizer.zero_grad()
        outputs = policy(inputs.flatten().unsqueeze(0))
        action =torch.argmax(outputs)
        # print(f"prdicted action {action} selected actions {labelss}")
        # print(f"Selected action {outputs} and target {labels.reshape(1,-1)}")
        loss = criterion(outputs, labels.reshape(1,-1))
        # print(f"We are working on Epoch {epoch} and loss is {loss}")
        loss_sum += loss
        losses.append(loss.detach().numpy())
        if (i)%batch_size ==0:
            print(f"loss is {loss_sum}")
            
            loss.backward()
            optimizer.step()

torch.save(policy.state_dict(
), "/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/LearnFromDomenstrations/checkpoints/policy.pt")


# Plot the loss over time
n = 50
returns = pd.Series(losses)
rolling_mean = returns.rolling(window=n).mean()
plt.plot(rolling_mean, label='Rolling Mean (window size {})'.format(n))

plt.plot(losses)
plt.xlabel("Training iteration")
plt.ylabel("Loss")

plt.show()
