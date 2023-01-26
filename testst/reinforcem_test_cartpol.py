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

########
from env import GridWorld
from read_env_json import read_env_sol_json

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', default=True, action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
# env.reset(seed=args.seed)
torch.manual_seed(args.seed)

mode = "train"
train_path = "/home/CE/musaeed/rl_assignments/test/task/task"
train_target_path = "/home/CE/musaeed/rl_assignments/test/task/solution"
# m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
# terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
actions = ['m', 'l', 'r', 'f', 'pick', 'put']
initial_settings = read_env_sol_json(mode, train_path, train_target_path)
print(f"{initial_settings[0]} \n\n best action seq {initial_settings[1]}")
# env = GridWorld(*initial_settings[0])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    # print(f"the probs are {probs}")
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 0
    action_seq = []
    eps_actions = []
    counter = []
    numEpisodes = 10000
    # for i_episode in count(1):
    for i_episode in range(numEpisodes):
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state.flatten())
            actionString = actions[action]
            # action = actionString
            state, reward, done, _ = env.step(action)
            # print(f"{state} and action {action}")
            # print(action)
            # if args.render:
            #     env.render()
            policy.rewards.append(reward)
            

            eps_actions.append(actionString)
            if done and reward ==0:
                print("we are here")
                action_seq.append(eps_actions)
                counter.append(i_episode)
                print(f"the action sequence is {eps_actions}")
            if done:
                
                break
          

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
        # if reward >= 0:

            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            print(f"the action sequence is {eps_actions}")
            break
    
    with open("/home/muhammed-saeed/Documents/rl_assignments/test/pg_reinforce/results.txt", "w") as fb:
        for number, solution in enumerate(action_seq):
            fb.write(f"The solution occured at {counter[number]} episode \n")
            fb.write(" ".join(solution))
            fb.write('\n-------------- -------------- ---------- -------- \n')

if __name__ == '__main__':
    main()