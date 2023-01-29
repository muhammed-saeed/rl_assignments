


import argparse
import gym
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
########
from env import GridWorld
from read_env_json import read_env_sol_json
from memory import get_memory

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
args = parser.parse_args()



# print(task[0])
# print(solution[0])

torch.manual_seed(args.seed)

mode = "train"
train_path = "/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/task/task"
train_target_path = "/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/task/solution"
# m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
# terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
actions = ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.affine2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.affine3 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = self.dropout(x)
        x = F.relu(x)

        action_scores = self.affine3(x)
        # action_scores = action_scores.unsqueeze(-1)

        print(action_scores)
        return F.log_softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()


    
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # print(state)
    probs = policy(state)
    # print(f"the probs are {probs} {probs.sum()}")
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

horizon = 100
task = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_medium/train/task"
seq = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_medium/train/seq"

tasks, optimumSolution, files = read_env_sol_json("train", task, seq)
print(f"{len(tasks)} and {len(optimumSolution)}")
memory = get_memory("train", task, seq)


def main():
    
    numTasks = len(memory)  # number of tasks in the memory
    for task in range(numTasks):
        for episode in memory[task]:
            state = episode[0]  # get the state from the memory
            action = episode[1]  # get the action from the memory
            agent_action = select_action(state)
            # state = torch.from_numpy(state).float()
            optimizer.zero_grad()
            log_probs = policy(state.flatten())  # pass the state through the policy network
            loss = loss_fn(log_probs, torch.tensor([action]))  # compute the loss
            loss.backward()  # backpropagate the gradients
            optimizer.step()  # update the policy network's parameters

        
if __name__ == '__main__':
    main()