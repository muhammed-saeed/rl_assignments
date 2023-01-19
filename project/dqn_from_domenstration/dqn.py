from env_4 import GridWorld
import numpy as np
import matplotlib.pyplot as plt

def maxAction(Q, state, actions):
    #this belongs to the agent
    #the agent estimates of the present value for each state, action pair
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    # map magic squares to their connecting square
    # magicSquares = {18: 54, 63: 14}
    #the first magic square is at position 18 and takes the agent to position54
    #the second magic square is at position 63 and takes the agent into position 14

    # m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
    #terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
    env = GridWorld(6, 6,(3,1),"west",[(6,0),(2,1)], [(3,2),(4,3)],((6,2),'east',[(2,1)]), ['m', 'l', 'r', 'f','pick','put'])
    env = GridWorld(4,4, (1,1), "west",[None],[(1,2),(2,3)],[(3,2),"east",[None]],['m', 'l', 'r', 'f'])
    # model hyperparameters
    ALPHA = 0.1
    #learning rate
    GAMMA = 1.0
    #discount factor
    EPS = 1.0
    #epsilon-greedy action selection "start random and then ends with selecting the best state-action pairs"


    Q = {}
    action_seq = []
    counter = []
    for state in env.stateSpacePlus:
        #for all possible states in the system make the action value function zero
        for action in env.possibleActions:
            Q[state, action] = 0
            #for start makes all the q-value "estimations to" zero
            #so if the agent is starting with zero, but then the agent go reward -1 which is significanly
            #worse than 0 then the agent think about doing more exploration.

    numGames = 10_000
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
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
                        GAMMA*Q[observation_,action_] - Q[observation,action])
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