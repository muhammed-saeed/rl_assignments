import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from env import GridWorld
from read_env_json import read_env_sol_json
import base64
from collections import defaultdict

symbols_b64 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'

def def_value():
    return 0

def v2r(n, base=symbols_b64): # value to representation
    """
    Convert a positive integer to its string representation in a custom base.
    
    :param n: the numeric value to be represented by the custom base
    :param base: the custom base defined as a string of characters, used as symbols of the base
    :returns: the string representation of natural number n in the custom base
    """
    if n == 0: return base[0]
    b = len(base)
    digits = ''
    while n > 0:
        digits = base[n % b] + digits
        n  = n // b
    return digits

def stringify_state(state):
    # print(state)
    str_state = ""
    for i in np.ravel(state):
        n = int(i)
        rep = v2r(n)
        # print(n, rep)
        str_state += rep
    return str_state

def maxAction(Q, state, actions):
    #this belongs to the agent
    #the agent estimates of the present value for each state, action pair
    state_str = stringify_state(state)
    values = np.array([Q[state_str,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    mode = "train"
    train_path = "/home/muhammed-saeed/Documents/rl_assignments/test/task/"
    train_target_path = "/home/muhammed-saeed/Documents/rl_assignments/test/solution/"
    # m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
    #terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
    actions = ['m', 'l', 'r', 'f','pick','put']
    initial_settings = read_env_sol_json(mode, train_path, train_target_path)
    print(f"initial settings are {initial_settings}")
    # print(f"{initial_settings[0]} \n\n {initial_settings[1]}")
    # env = GridWorld(*initial_settings[0])
    #the * to seperate the elements of the array
    env = GridWorld(4,4, (1,1), "west",[],[(1,2),(2,3)],[(3,2),"east",[]],['m', 'l', 'r', 'f', "put", "pick"])
    # model hyperparameters
    ALPHA = 0.1
    #learning rate
    GAMMA = 1.0
    #discount factor
    EPS = 1.0
    #epsilon-greedy action selection "start random and then ends with selecting the best state-action pairs"


    Q = defaultdict(def_value)
    action_seq = []
    counter = []
    for state in env.stateSpacePlus:
        #for all possible states in the system make the action value function zero
        for action in env.possibleActions:
            Q[state, action] = 0
            #for start makes all the q-value "estimations to" zero
            #so if the agent is starting with zero, but then the agent go reward -1 which is significanly
            #worse than 0 then the agent think about doing more exploration.

    numGames = 100_000
    totalRewards = np.zeros(numGames)
    env.render()
    for i in range(numGames):
        eps_actions = []
        if i % 100 == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        observation = env.reset()
        #at the beginning of each episode with your environment then reset the system
        while not done:
            rand = np.random.random()
            action = maxAction(Q,observation, env.possibleActions) if rand < (1-EPS) \
                                                    else env.actionSpaceSample()
            observation_, reward, done, info = env.step(action)
            
            eps_actions.append(action)
            if done and reward ==0:
                action_seq.append(eps_actions)
                counter.append(i)
                print(eps_actions)
            #take the action from the maxAction
            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)
            # print(observation)
            obs_str = stringify_state(observation)
            obs_str_ = stringify_state(observation_)
            Q[obs_str,action] = Q[obs_str,action] + ALPHA*(reward + \
                        GAMMA*Q[obs_str_,action_] - Q[obs_str,action])
            #calculation the max_action for the new_state, this where the alpha comes in to modify the values
            #
            observation = observation_
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards
    with open("/home/muhammed-saeed/Documents/rl_assignments/assingment_4/results.txt","w") as fb:
        for number, solution in enumerate(action_seq):
            fb.write(f"The solution occured at {counter[number]} episode \n")
            fb.write(" ".join(solution))
            fb.write('\n-------------- -------------- ---------- -------- \n')
    plt.plot(totalRewards)
    plt.show()