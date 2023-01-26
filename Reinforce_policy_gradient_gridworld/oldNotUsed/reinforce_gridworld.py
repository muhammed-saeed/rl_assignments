
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from Reinforce_policy_gradient_gridworld.env import GridWorld
from read_env_json import read_env_sol_json


def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


l1 = 32  # since 4*4 grid
l2 = 150
l3 = 150
l4 = 6  # number of actions


model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l3, l4),
    torch.nn.Softmax(dim=0)  # C
)

learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# A Input data is length 4
# B Output is a 2-length vector for the Left and the Right actions
# C Output is a softmax probability distribution over actions


def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards  # A
    disc_return /= disc_return.max()  # B
    return disc_return

# A Compute exponentially decaying rewards
# B Normalize the rewards to be within the [0,1] interval to improve numerical stability


def loss_fn(preds, r):  # A
    return -1 * torch.sum(r * torch.log(preds))  # B

# A The loss function expects an array of action probabilities for the actions that were taken and the discounted rewards.
# B It computes the log of the probabilities, multiplies by the discounted rewards, sums them all and flips the sign.

# env = gym.make("CartPole-v0")


mode = "train"
train_path = "/home/muhammed-saeed/Documents/rl_assignments/test/task/task"
train_target_path = "/home/muhammed-saeed/Documents/rl_assignments/test/task/solution"
# m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
# terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
actions = ['m', 'l', 'r', 'f', 'pick', 'put']
initial_settings = read_env_sol_json(mode, train_path, train_target_path)
print(f"{initial_settings[0]} \n\n {initial_settings[1]}")
env = GridWorld(*initial_settings[0])


MAX_DUR = 1000
MAX_EPISODES = 10000
gamma = 0.99
score = []  # A
expectation = 0.0
action_seq = []
action_sequence = []
counter = []
for episode in range(MAX_EPISODES):
    curr_state = env.reset()
    # print(f"curr_state{curr_state}")
    # print(f"{len(curr_state.flatten())}")
    done = False
    transitions = []  # B

    eps_actions = []
    for t in range(MAX_DUR):  # C
        act_prob = model(torch.from_numpy(curr_state.flatten()).float())
        print(f"action_prob {act_prob}")
        action = np.random.choice(np.arange(6), p=act_prob.data.numpy())  # E
        prev_state = curr_state
        actionString = actions[action]
        action_sequence.append(actions[action])
        curr_state, reward, done, info = env.step(actionString)  # F
        print(f"{actionString} and {reward}")
        # transitions.append((prev_state, action, t+1)) #G
        transitions.append((prev_state, action, reward))  # G
        # env.render()
        # print(f"{prev_state} and {actionString} and {reward}")
        if done:  # H
            break
        if done and reward == 0:
            action_seq.append(eps_actions)
            counter.append(episode)
            print(eps_actions)

    ep_len = len(transitions)  # I
    score.append(ep_len)
    reward_batch = torch.Tensor(
        [r for (s, a, r) in transitions]).flip(dims=(0,))  # J
    disc_returns = discount_rewards(reward_batch)  # K
    state_batch = torch.Tensor([s.flatten() for (s, a, r) in transitions])  # L
    action_batch = torch.Tensor([a for (s, a, r) in transitions])  # M
    pred_batch = model(state_batch)  # N
    prob_batch = pred_batch.gather(
        dim=1, index=action_batch.long().view(-1, 1)).squeeze()  # O
    loss = loss_fn(prob_batch, disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with open("/home/muhammed-saeed/Documents/rl_assignments/ProjectSolution/ours/pg_reinforce/results.txt", "w") as fb:
    for number, solution in enumerate(action_seq):
        fb.write(f"The solution occured at {counter[number]} episode \n")
        fb.write(" ".join(solution))
        fb.write('\n-------------- -------------- ---------- -------- \n')

# A List to keep track of the episode length over training time
# B List of state, action, rewards (but we ignore the reward)
# C While in episode
# D Get the action probabilities
# E Select an action stochastically
# F Take the action in the environment
# G Store this transition
# H If game is lost, break out of the loop
# I Store the episode length
# J Collect all the rewards in the episode in a single tensor
# K Compute the discounted version of the rewards
# L Collect the states in the episode in a single tensor
# M Collect the actions in the episode in a single tensor
# N Re-compute the action probabilities for all the states in the episode


score = np.array(score)
avg_score = running_mean(score, 50)

plt.figure(figsize=(10, 7))
plt.ylabel("Episode Duration", fontsize=22)
plt.xlabel("Training Epochs", fontsize=22)
plt.plot(avg_score, color='green')
