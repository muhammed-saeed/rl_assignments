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
from memory import get_memory
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
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
# env.reset(seed=args.seed)
torch.manual_seed(args.seed)

mode = "train"
train_path = "/home/muhammed-saeed/Desktop/testst/task/task"
train_target_path = "/home/muhammed-saeed/Desktop/testst/task/solution"
# m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
# terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
actions = ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']
initial_settings = read_env_sol_json(mode, train_path, train_target_path)
print(f"{initial_settings[0]} \n\n") # best action seq {initial_settings[1]}")
# print("#############################")
# a = initial_settings[0]
# print(len(a))
# print(initial_settings[0])
env = GridWorld(*initial_settings[0])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.affine2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.affine3 = nn.Linear(128, 6)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x.requires_grads = True

        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = self.dropout(x)
        x = F.relu(x)

        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-5)
eps = np.finfo(np.float32).eps.item()
# eps = 1e-3
# eps = 1e-7


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # print(state)
    probs = policy(state)
    # print(f"the probs are {probs} {probs.sum()}")
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
    # print(finish_episodereturns)
    # print("Length", len(policy.saved_log_probs), len(returns))
    returns = torch.tensor(returns)
    # print("Return", returns, returns.mean(), returns.std(unbiased=False))
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + eps)
    # print("Return", returns)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        # print("Reward", R)
        policy_loss.append(-log_prob * R)
    # print(policy_loss, torch.cat(policy_loss).mean())

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


horizon = 50


mode = "train"

easy_tasks = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_easy/train/"
easy_solution = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_easy/seq/"

medium_tasks = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_medium/train/task/"
medium_solutions = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_medium/train/seq/"




memory = get_memory()
easy_memory = get_memory(mode, easy_tasks, easy_solution)

def curriculum_design():
    print(len(easy_memory))


def mainLearnFromDomenstrations():
    print(len(easy_memory))
    return
    running_reward = 0
    action_seq = []
    eps_actions = []
    counter = []
    numEpisodes = len(memory)
    # numEpisodes = 5_000
    for i_episode, EPISODE in enumerate(numEpisodes):
        eps_actions = []
        state = env.reset()
        ep_reward = 0
        test_reward = None
        last_action = None
        for t,episode in enumerate(EPISODE):  # Don't infinite loop while learning
            action = episode[1]
            actionString = actions[action]
            last_action = actionString
            # print(actionString)
            state, reward, done, _ = env.step(actionString)
            eps_actions.append(actionString)

            # if not done and t+1 == horizon:
            #     # print('hi reasched here !!!!')
            #     # print(eps_actions)
            #     # reward = env.crashReward
            #     done = True

            policy.rewards.append(reward)
            ep_reward += reward
            test_reward = reward
            if done and reward ==env.winReward:
                print("we are here")
                action_seq.append(eps_actions)
                counter.append(i_episode)
                print(f"the action sequence is {eps_actions}")
            if done:
                break
          

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {test_reward}\tAverage reward: {running_reward} \t Last action {last_action}') 
            #.format(      i_episode, test_reward, running_reward))
            torch.save(policy.state_dict(),"/home/muhammed-saeed/Desktop/testst/gridworld/policy_log_interval.pt")
        # if running_reward > env.spec.reward_threshold:
        if reward>=0 and reward <env.winReward:
            print("reward design")
        if reward >= env.winReward:

            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            print(f"the action sequence is {eps_actions}")
            torch.save(policy.state_dict(),"/home/muhammed-saeed/Desktop/testst/gridworld/policy_solved_task.pt")
            with open("/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/results.txt", "w") as fb:
                for number, solution in enumerate(action_seq):
                    fb.write(f"The solution occured at {counter[number]} episode \n")
                    fb.write(" ".join(solution))
                    fb.write('\n-------------- -------------- ---------- -------- \n')

    print("Saving the final weights after leanring from domenstraiton over the entire domenstrationData")
    torch.save(policy.state_dict(),"/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/learnFromDomenstration/LFD_policy_network.pt")

def main():
    print(len(memory))
    
    running_reward = 0
    action_seq = []
    eps_actions = []
    counter = []
    numEpisodes = 1_000_000
    numEpisodes = 5_000
    for i_episode in range(numEpisodes):
        eps_actions = []
        state = env.reset()
        ep_reward = 0
        test_reward = None
        last_action = None
        for t in range(horizon):  # Don't infinite loop while learning
            action = select_action(state.flatten())
            actionString = actions[action]
            last_action = actionString
            # print(actionString)
            state, reward, done, _ = env.step(actionString)
            eps_actions.append(actionString)

            if not done and t+1 == horizon:
                # print('hi reasched here !!!!')
                # print(eps_actions)
                # reward = env.crashReward
                done = True

            policy.rewards.append(reward)
            ep_reward += reward
            test_reward = reward
            if done and reward ==env.winReward:
                print("we are here")
                action_seq.append(eps_actions)
                counter.append(i_episode)
                print(f"the action sequence is {eps_actions}")
            if done:
                break
          

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {test_reward}\tAverage reward: {running_reward} \t Last action {last_action}') 
            #.format(      i_episode, test_reward, running_reward))
            torch.save(policy.state_dict(),"/home/muhammed-saeed/Desktop/testst/gridworld/policy_log_interval.pt")
        # if running_reward > env.spec.reward_threshold:
        if reward>=0 and reward <env.winReward:
            print("reward design")
        if reward >= env.winReward:

            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            print(f"the action sequence is {eps_actions}")
            torch.save(policy.state_dict(),"/home/muhammed-saeed/Desktop/testst/gridworld/policy_solved_task.pt")
            with open("/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/results.txt", "w") as fb:
                for number, solution in enumerate(action_seq):
                    fb.write(f"The solution occured at {counter[number]} episode \n")
                    fb.write(" ".join(solution))
                    fb.write('\n-------------- -------------- ---------- -------- \n')
            # break
    
    # with open("/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/results.txt", "w") as fb:
    #     for number, solution in enumerate(action_seq):
    #         fb.write(f"The solution occured at {counter[number]} episode \n")
    #         fb.write(" ".join(solution))
    #         fb.write('\n-------------- -------------- ---------- -------- \n')

if __name__ == '__main__':
    # main()
    mainLearnFromDomenstrations()