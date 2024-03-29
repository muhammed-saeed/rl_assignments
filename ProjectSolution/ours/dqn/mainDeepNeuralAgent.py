import sys
import os 
import torch
import random
import numpy as np
import matplotlib as plt
from env import GridWorld
from collections import deque
from read_env_json import read_env_sol_json

# from dqnAgent import model

import copy

l1 = 64
l2 = 150
l3 = 100
l4 = 4


model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)

model2 = copy.deepcopy(model) #A
model2.load_state_dict(model.state_dict()) #B

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9
epsilon = 0.3


mode = "train"
train_path = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data/train/task"
train_target_path = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data/train/seq"
# m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
#terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
actions = ['m', 'l', 'r', 'f','pick','put']
memory = read_env_sol_json(mode, train_path, train_target_path)
print(memory[0])
# memory = get_memory(mode, train_path, train_target_path)
# print(f"{initial_settings[0]} \n\n {initial_settings[1]}")
# env = GridWorld(*initial_settings[0])

### up until here is working

epochs = 100
losses = []
mem_size = 1000 #A
batch_size = 200 #B
replay = deque(maxlen=mem_size) #C
max_moves = 50 #D
h = 0



from collections import deque
epochs = 5000
losses = []
mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size)
max_moves = 50
h = 0
sync_freq = 500 #A
j=0
for i in range(epochs):
    # env = GridWorld(*initial_settings[0])
    # state1_ = env.reset()
    # state1 = torch.from_numpy(state1_).float()
    # status = 1
    # mov = 0
    # while(status == 1): 
        # j+=1
        # mov += 1
        # qval = model(state1)
        # qval_ = qval.data.numpy()
        # if (random.random() < epsilon):
        #     action_ = np.random.randint(0,4)
        # else:
        #     action_ = np.argmax(qval_)
        
        # action = action_set[action_]
        # game.makeMove(action)
        # state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        # state2 = torch.from_numpy(state2_).float()
        # reward = game.reward()
        # done = True if reward > 0 else False
        # exp =  (state1, action_, reward, state2, done)
        # replay.append(exp) #H
        # state1 = state2
        
        # if len(replay) > batch_size:
            minibatch = random.sample(memory, batch_size)
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            Q1 = model(state1_batch) 
            with torch.no_grad():
                Q2 = model2(state2_batch) #B
            
            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            # clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            # if j % sync_freq == 0: #C
            #     model2.load_state_dict(model.state_dict())
        # if reward != -1 or mov > max_moves:
        #     status = 0
        #     mov = 0
        
losses = np.array(losses)

plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)
#A Set the update frequency for synchronizing the target model parameters to the main DQN
#B Use the target network to get the maiximum Q-value for the next state
#C Copy the main model parameters to the target network

#A Set the total size of the experience replay memory
#B Set the minibatch size
#C Create the memory replay as a deque list
#D Maximum number of moves before game is over
#E Compute Q-values from input state in order to select action
#F Select action using epsilon-greedy strategy
#G Create experience of state, reward, action and next state as a tuple
#H Add experience to experience replay list
#I If replay list is at least as long as minibatch size, begin minibatch training
#J Randomly sample a subset of the replay list
#K Separate out the components of each experience into separate minibatch tensors
#L Re-compute Q-values for minibatch of states to get gradients
#M Compute Q-values for minibatch of next states but don't compute gradients
#N Compute the target Q-values we want the DQN to learn
#O If game is over, reset status and mov number