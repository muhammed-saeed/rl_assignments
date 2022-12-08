import numpy as np
import matplotlib.pyplot as plt

# the concept of the state-space all states- excluding the termnial state
# main loop with the agent in which we are using q_learning

class GridWorld(object):
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m,n))
        #the empty states are represented by one 
        #the agent is representented by zero
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m*self.n)]
        #list comperhension for all states in the sytem
        self.stateSpace.remove(80)
        #remove the terminal state from the list
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        #define the next statespace
        self.actionSpace = {'U': -self.m, 'D': self.m,
                            'L': -1, 'R': 1}
        #moving up will reduce the agent position one row so m
        #movnig down will advance the agent position one rown so m states
        self.possibleActions = ['U', 'D', 'L', 'R']
        #group of possible actions


        # dict with magic squares and resulting squares
        self.addMagicSquares(magicSquares)
        self.agentPosition = 0

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace
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

    numGames = 10
    totalRewards = np.zeros(numGames)
    env.render()
    for i in range(numGames):
        if i % 2 == 0:
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