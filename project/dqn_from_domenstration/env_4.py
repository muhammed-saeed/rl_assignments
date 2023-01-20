import numpy as np
import matplotlib.pyplot as plt

# the concept of the state-space all states- excluding the termnial state
# main loop with the agent in which we are using q_learning

class GridWorld(object):
    def __init__(self, m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
        self.grid = np.zeros((m,n))
        self.grid[init_state] = 1
        #the location of the agent to be 1
        self.gridRows = m
        self.gridCols = n
        #the empty states are represented by one 
        #the agent is representented by zero
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m*self.n)]
        #list comperhension for all states in the sytem
        


        
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        #define the next statespace
        self.agentOrientation = orientation


        self.initial_wall_locations = list(wall_locations)
        self.initiial_markers_locations = list(markers_locations)
        self.init_orientation = orientation
        self.orientation = orientation
        self.init_state = init_state

        #moving up will reduce the agent position one row so m
        #movnig down will advance the agent position one rown so m states
        self.agentisAlive =  True
        self.isHandEmpty = True
    
        self.markers_locations = list(markers_locations)
        self.wallLocations = list(wall_locations)
        for wall in self.wallLocations:
            self.stateSpace.remove(wall[0]*self.n + wall[1])
        #remove the terminal state from the list
        self.terminalStates = terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]

        #
        self.possibleActions = possible_actions
       
        self.agentPosition = self.n*init_state[0] + init_state[1]
        self.init_position = self.agentPosition
        self.crashReward = -1000
        self.reward = -1
     
    def actionSpace(self,action):
            #the move action
            if action == 'm':
                if self.orientation == "north":
                    return -self.m, self.reward
                elif self.orientation == "south":
                    return +self.m, self.reward
                elif self.orientation == "west":
                    return -1, self.reward
                elif self.orientation == "east":
                    return 1, self.reward

            #the left action
            elif action == "l":
                if self.orientation == "south":
                    self.orientation = "east"
                elif self.orientation == "north":
                    self.orientation = "west"
                elif self.orientation == "east":
                    self.orientation = "north"
                elif self.orientation == "west":
                    self.orientation = "south"
                return 0, self.reward
            
            #the right action
            elif action == "r":
                if self.orientation == "south":
                    self.orientation = "west"
                elif self.orientation == "north":
                    self.orientation = "east"
                elif self.orientation == "east":
                    self.orientation = "south"
                elif self.orientation == "west":
                    self.orientation = "north"
                return 0, self.reward

            #the pickup action
            elif action == "pick":
                if self.state_is_marker():
                        self.isHandEmpty = False
                        #take the marker from the box
                        x,y = self.getAgentRowAndColumn()
                        
                        # we want to remove this marker, then we count the number of states we have in the system
                        marker_state = x*self.gridRows + y
                        markers_states = [x*self.gridRows + y for x,y in self.markers_locations]
                        index_ = markers_states.index(marker_state)
                        self.markers_locations.pop(index_)

                        return 0, self.reward
                else: 
                    self.agentisAlive = False
                    return 0, self.crashReward 
                    #if the agent crashes then the reward has to be high negatve value
                
            elif action == "put":
                if not self.isHandEmpty:
                    x,y = self.getAgentRowAndColumn()

                    self.markers_locations.append((x,y))
                    return 0, self.reward
                else:
                    self.agentisAlive = False
                    return 0, self.crashReward


            
            elif action == "f":
                if self.isTerminalState():
                    #how to define that the agent_is in the goal state anyways?!
                    self.agentisAlive = True
                    return 0, 0 #give the agent large reward "0" while negative else of reaching the final state
                else:
                    self.agentisAlive = False
                    return 0, self.crashReward
                
            



    
    def state_is_marker(self):
        if (self.getAgentRowAndColumn()) in self.markers_locations:
            return True

    def isTerminalState(self):
        # return state in self.stateSpacePlus and state not in self.stateSpace
        goal_state = self.terminalStates[0]
        goal_orientation = self.terminalStates[1]
        goal_markers = self.terminalStates[2]
        return self.getAgentXY(self.agentPosition ) == goal_state and self.orientation == goal_orientation and set(self.markers_locations) == set(goal_markers)

    
    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y


    def getAgentXY(self, state):
        x = state // self.m
        y = state % self.n
        return x, y
    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        #but here if the agent is top-left and moved left it get reward of -1 and still ther
        #I want the agent to be dead
    
        self.grid[x][y] = 0
        #so assign empty-state to the agent old position
        self.agentPosition = state
        #assign the agent new position to 1
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1
        #put the agent position as 1

    def offGridMove(self, changeState):
        # if we move into a row not in the grid
        
        x,y = self.getAgentRowAndColumn()
        if self.orientation == "north" and x ==0:
            #if the new state is not in the grid world then we are moving out of the grid
            return True
        # if we're trying to wrap around to next row
        if self.orientation == "south" and x==self.m-1:
            #if the new state is not in the grid world then we are moving out of the grid
            return True
        
        if self.orientation == "east" and y==self.n-1:
            #if the new state is not in the grid world then we are moving out of the grid
            return True
        
        if self.orientation == "west" and y==0:
            #if the new state is not in the grid world then we are moving out of the grid
            return True

        ### walls
        newState = self.agentPosition + changeState
        x,y = self.getAgentXY(newState)
        return (x,y) in self.wallLocations
        

    def step(self, action):
        agentX, agentY = self.getAgentRowAndColumn()
        stateChange, REWARD = self.actionSpace(action)
        offGridBool= self.offGridMove(stateChange)
        if  offGridBool:
            self.agentisAlive = False
            return self.agentPosition, self.crashReward, True, None
            #since the agent already is outside the grid and died
        else:
            self.agentPosition += stateChange
            resultingState = self.agentPosition
            self.setState(resultingState)

            done = REWARD==0 or not self.agentisAlive
            return resultingState, REWARD, done, None
            #the only scenario in which rewards is zero is when the agent finish using finish.

    


    def reset(self):
        self.agentPosition = self.init_position
        self.agentisAlive = True
        self.grid = np.zeros((self.m,self.n))
        self.markers_locations = self.initiial_markers_locations
        self.wallLocations = self.initial_wall_locations
        self.orientation = self.init_orientation
        self.grid = np.zeros((self.m,self.n))
        self.grid[self.init_state] = 1
        return self.agentPosition

    def render(self):
        print('------------------------------------------')
        for row in range(self.m):
            for col in range(self.n):
                if self.grid[row][col] == 0:
                    if (row,col) in self.wallLocations: 
                        print("W", end="\t")
                    elif (row,col) in self.markers_locations:
                        print("M",end="\t")
                    else:
                        

                        print('-', end='\t')
                else:
                    #col=1 then this mean we have an-agent there and then print X
                    print('X', end='\t')
                
            print('\n')
            #after each line we want to print a new line
        print('------------------------------------------')

    def actionSpaceSample(self):
        #return just a random choice of list of possible options in the system
        return np.random.choice(self.possibleActions)



# if __name__ == '__main__':
#     # map magic squares to their connecting square
#     # magicSquares = {18: 54, 63: 14}
#     #the first magic square is at position 18 and takes the agent to position54
#     #the second magic square is at position 63 and takes the agent into position 14

#     # m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
#     #terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
#     env = GridWorld(6, 6,(3,1),"west",[(6,0),(2,1)], [(3,2),(4,3)],((6,2),'east',[(2,1)]), ['m', 'l', 'r', 'f','pick','put'])
#     env = GridWorld(4,4, (1,1), "west",[None],[(1,2),(2,3)],[(3,2),"east",[None]],['m', 'l', 'r', 'f'])
#     # model hyperparameters
#     ALPHA = 0.1
#     #learning rate
#     GAMMA = 1.0
#     #discount factor
#     EPS = 1.0
#     #epsilon-greedy action selection "start random and then ends with selecting the best state-action pairs"


#     Q = {}
#     action_seq = []
#     counter = []
#     for state in env.stateSpacePlus:
#         #for all possible states in the system make the action value function zero
#         for action in env.possibleActions:
#             Q[state, action] = 0
#             #for start makes all the q-value "estimations to" zero
#             #so if the agent is starting with zero, but then the agent go reward -1 which is significanly
#             #worse than 0 then the agent think about doing more exploration.

#     numGames = 10_000
#     totalRewards = np.zeros(numGames)
#     env.render()
#     for i in range(numGames):
#         eps_actions = []
#         if i % 100 == 0:
#             print('starting game ', i)
#         done = False
#         epRewards = 0
#         observation = env.reset()
#         #at the beginning of each episode with your environment then reset the system
#         while not done:
#             rand = np.random.random()
#             action = maxAction(Q,observation, env.possibleActions) if rand < (1-EPS) \
#                                                     else env.actionSpaceSample()
#             observation_, reward, done, info = env.step(action)
            
#             eps_actions.append(action)
#             if done and reward ==0:
#                 action_seq.append(eps_actions)
#                 counter.append(i)
#                 print(eps_actions)
#             #take the action from the maxAction
#             epRewards += reward

#             action_ = maxAction(Q, observation_, env.possibleActions)
#             Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
#                         GAMMA*Q[observation_,action_] - Q[observation,action])
#             #calculation the max_action for the new_state, this where the alpha comes in to modify the values
#             #
#             observation = observation_
#         if EPS - 2 / numGames > 0:
#             EPS -= 2 / numGames
#         else:
#             EPS = 0
#         totalRewards[i] = epRewards
#     with open("/home/muhammed-saeed/Documents/rl_assignments/assingment_4/results.txt","w") as fb:
#         for number, solution in enumerate(action_seq):
#             fb.write(f"The solution occured at {counter[number]} episode \n")
#             fb.write(" ".join(solution))
#             fb.write('\n-------------- -------------- ---------- -------- \n')
#     plt.plot(totalRewards)
#     plt.show()