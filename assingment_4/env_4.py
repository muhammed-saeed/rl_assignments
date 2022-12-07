import numpy as np
import matplotlib.pyplot as plt

# the concept of the state-space all states- excluding the termnial state
# main loop with the agent in which we are using q_learning

class GridWorld(object):
    def __init__(self, m, n, magicSquares, markers_locations, wall_locations, terminal_state):
        self.grid = np.zeros((m,n))
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
        self.agentOrientation = "south"
        # self.actionSpace = {'m': -self.m, 'r': self.m,
        #                     'l': -1, 'p': 1, 'f': 0}

        #moving up will reduce the agent position one row so m
        #movnig down will advance the agent position one rown so m states
        self.agentisAlive =  True
        self.isHandEmpty = True
        self.markers_locations = markers_locations
        self.wallLocations = wall_locations
        for wall in self.wallLocations:
            self.stateSpace.remove(80)
        #remove the terminal state from the list
        self.terminalStates = terminal_state
        self.possibleActions = ['m', 'D', 'L', 'R','f']
        #group of possible actions


        # dict with magic squares and resulting squares
        # self.addMagicSquares(magicSquares)
        self.agentPosition = 0

     
    def actionSpace(self,action):
            #the move action
            if action == 'm':
                if self.orientation == "north":
                    return -self.m
                elif self.orientation == "south":
                    return +self.m
                elif self.orientation == "west":
                    return -1
                elif self.orientation == "east":
                    return 1

            #the left action
            elif action == "l":
                if self.orientation == "south":
                    self.orientation = "east"
                elif self.orienation == "north":
                    self.orientation = "west"
                elif self.orientation == "east":
                    self.orientation = "north"
                elif self.orinetation == "west":
                    self.orientation = "south"
            
            #the right action
            elif action == "r":
                if self.orientation == "south":
                    self.orientation = "west"
                elif self.orienation == "north":
                    self.orientation = "east"
                elif self.orientation == "east":
                    self.orientation = "south"
                elif self.orinetation == "west":
                    self.orientation = "north"

            #the pickup action
            elif action == "pick":
                if self.state_is_marker():
                        self.isHandEmpty = False
                        #take the marker from the box
                        x,y = self.getAgentRowAndColumn()
                        
                        # we want to remove this marker, then we count the number of states we have in the system
                        marker_state = x*self.gridRows + y
                        markers_states = [x*self.gridRows + y for [x,y] in self.markers_locations]
                        index_ = markers_states.index(marker_state)
                        self.markers_locations.pop(index_)

                        return self.reward
                else: 
                    self.agentisAlive = False
                    return self.agentisAlive 
                    #if the agent crashes then the reward has to be high negatve value
                
            elif action == "put":
                if not self.isHandEmpty:
                    x,y = self.getAgentRowAndColumn()

                    self.markers_locations.append([x,y])


            
            elif action == "f":
                if self.state  in self.goal_states:
                    #how to define that the agent_is in the goal state anyways?!
                    self.isAlive = False
                    return 0 #give the agent large reward "0" while negative else of reaching the final state

                
            



    
    def state_is_marker(self):
        if np.array(self.getAgentRowAndColumn()) in self.markers_locations:
            return True

    def isTerminalState(self, state):
        # return state in self.stateSpacePlus and state not in self.stateSpace
        if self.getAgentRowAndColumn 
        #the differnece betweent the stateSpacePlus and the stateSpace are the terminalstate


    def addMagicSquares(self, magicSquares):
        self.magicSquares = magicSquares
        i = 2
        #the representation of the magic square in the grid world to be 2
        for square in self.magicSquares:
            x = square // self.m
            #we need to know what position we are in
            #the x will be the floor of the current squre/rows
            y = square % self.n
            #y is the modules of the columns
            self.grid[x][y] = i
            i += 1
            x = magicSquares[square] // self.m
            y = magicSquares[square] % self.n
            self.grid[x][y] = i
            i += 1

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
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

    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpacePlus:
            #if the new state is not in the grid world then we are moving out of the grid
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.m == 0 and newState  % self.m == self.m - 1:
            #if you are in the bottom one and you want to reduce it so reach in the top row
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False

    def step(self, action):
        agentX, agentY = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]
        #the actionSpace return the change in the state that induced when perfomring the specific action

        if resultingState in self.magicSquares.keys():
            resultingState = magicSquares[resultingState]

        reward = -1 if not self.isTerminalState(resultingState) else 0
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            #open-ai gym return the new-state, the reward, the game is over, more information
            return resultingState, reward, \
                   self.isTerminalState(resultingState), None
        else:
            return self.agentPosition, reward, \
                   self.isTerminalState(self.agentPosition), None

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m,self.n))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition

    def render(self):
        print('------------------------------------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    #col=1 then this mean we have an-agent there and then print X
                    print('X', end='\t')
                elif col == 2:
                    #if the column is 2 then this is onw of of our magic squares entrace
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                elif col == 5:
                    print('Bout', end='\t')
            print('\n')
            #after each line we want to print a new line
        print('------------------------------------------')

    def actionSpaceSample(self):
        #return just a random choice of list of possible options in the system
        return np.random.choice(self.possibleActions)

def maxAction(Q, state, actions):
    #this belongs to the agent
    #the agent estimates of the present value for each state, action pair
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    # map magic squares to their connecting square
    magicSquares = {18: 54, 63: 14}
    #the first magic square is at position 18 and takes the agent to position54
    #the second magic square is at position 63 and takes the agent into position 14
    env = GridWorld(9, 9, magicSquares)
    # model hyperparameters
    ALPHA = 0.1
    #learning rate
    GAMMA = 1.0
    #discount factor
    EPS = 1.0
    #epsilon-greedy action selection "start random and then ends with selecting the best state-action pairs"


    Q = {}
    for state in env.stateSpacePlus:
        #for all possible states in the system make the action value function zero
        for action in env.possibleActions:
            Q[state, action] = 0
            #for start makes all the q-value "estimations to" zero
            #so if the agent is starting with zero, but then the agent go reward -1 which is significanly
            #worse than 0 then the agent think about doing more exploration.

    numGames = 50000
    totalRewards = np.zeros(numGames)
    env.render()
    for i in range(numGames):
        if i % 5000 == 0:
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

    plt.plot(totalRewards)
    plt.show()