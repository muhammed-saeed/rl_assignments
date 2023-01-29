import numpy as np
import copy
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
        

        self.initState = init_state
        
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        #define the next statespace
        self.agentOrientation = orientation


        self.initial_wall_locations = copy.deepcopy(wall_locations)
        self.initiial_markers_locations = copy.deepcopy(markers_locations)
        self.init_orientation = orientation
        self.orientation = orientation
        self.init_state = init_state

        #moving up will reduce the agent position one row so m
        #movnig down will advance the agent position one rown so m states
        self.agentisAlive =  True
        self.isHandEmpty = False
    
        self.markers_locations = copy.deepcopy(markers_locations)
        self.wallLocations = copy.deepcopy(wall_locations)
        for wall in self.wallLocations:
            self.stateSpace.remove(wall[0]*self.n + wall[1])
        #remove the terminal state from the list
        self.terminalStates = terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]

        #
        self.possibleActions = possible_actions
       
        self.agentPosition = self.n*init_state[0] + init_state[1]
        self.init_position = self.agentPosition
        self.winReward = 10000
        self.crashReward = -1000
        self.reward = -10
        self.rewardDesginVal = 100
        
        # print  ("!!!!!ENV CHECK!!!!!")
        # print  (f"{self.m},{self.n}, ({self.init_state}), {self.orientation}, ({self.initiial_markers_locations}), ({self.initial_wall_locations}), (({self.terminalStates})")
        # print  ("!!!!!ENV CHECK!!!!!")
    def actionSpace(self,action):
            #the move action
            if action == 'move':
                if self.orientation == "north":
                    return -self.m, self.reward
                elif self.orientation == "south":
                    return +self.m, self.reward
                elif self.orientation == "west":
                    return -1, self.reward
                elif self.orientation == "east":
                    return 1, self.reward

            #the left action
            elif action == "left":
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
            elif action == "right":
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
            elif action == "pickMarker":
                if self.state_is_marker():
                        # self.isHandEmpty = False
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
                
            elif action == "putMarker":
                # if not self.isHandEmpty:
                    x,y = self.getAgentRowAndColumn()
                    # print (f"({x},{y})")
                    # print (self.markers_locations)
                    if (x,y) in self.markers_locations:
                        self.agentisAlive = False
                        return 0, self.crashReward
                        #prevent the agent from put the marker in state that already has marker on it
                    if (x,y) not in self.markers_locations:
                        # print  ("yeah already in the markers")
                        self.markers_locations.append((x,y))
                    isRewardDesign, rewardDesign = self.rewardDesign()
                    if isRewardDesign:
                        # print  (f"reward Design {rewardDesign}")

                        return 0,rewardDesign
                    else:
                        return 0, self.reward
                # else:
                #     self.agentisAlive = False
                #     return 0, self.crashReward
                    # print (self.markers_locations)


            
            elif action == "finish":
                if self.isTerminalState():
                    #how to define that the agent_is in the goal state anyways?!
                    self.agentisAlive = True
                    return 0, self.winReward #give the agent large reward "0" while negative else of reaching the final state
                else:
                    self.agentisAlive = False
                    return 0, self.crashReward
                
            



    
    def state_is_marker(self):
        if (self.getAgentRowAndColumn()) in self.markers_locations:
            # print  (f"#################### \n Agent locaiton is {self.getAgentXY(self.agentPosition)} and markers {self.markers_locations}")
            return True
        else:
            return False

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
        

    def rewardDesign(self):
        return set(self.markers_locations) == set(self.terminalStates[2]), self.rewardDesginVal
            

    def step(self, action):
        agentX, agentY = self.getAgentRowAndColumn()
        stateChange, REWARD = self.actionSpace(action)
        if action == "move":
            offGridBool= self.offGridMove(stateChange)
            if  offGridBool:
                self.agentisAlive = False
                return self.agentPosition, self.crashReward, True, None
                #since the agent already is outside the grid and died
        # else:
        self.agentPosition += stateChange
        resultingState = self.agentPosition
        self.setState(resultingState)
        resultingState = self.get_state()
        done = REWARD==self.winReward or not self.agentisAlive
        return resultingState, REWARD, done, None
        #the only scenario in which rewards is 10000 is when the agent finish using finish.

    


    def reset(self):
        self.agentPosition = self.init_position
        self.agentisAlive = True
        self.grid = np.zeros((self.m,self.n))
        self.markers_locations = copy.deepcopy(self.initiial_markers_locations)
        self.wallLocations = copy.deepcopy(self.initial_wall_locations)
        self.orientation = self.init_orientation
        self.grid = np.zeros((self.m,self.n))
        self.grid[self.init_state] = 1
        # return self.agentPosition
        return self.get_state()

    def render(self):
        # print ('------------------------------------------')
        for row in range(self.m):
            for col in range(self.n):
                if self.grid[row][col] == 0:
                    if (row,col) in self.wallLocations: 
                        print ("W", end="\t")
                    elif (row,col) in self.markers_locations:
                        print ("move",end="\t")
                    else:
                        

                        print ('-', end='\t')
                else:
                    #col=1 then this mean we have an-agent there and then # print  X
                    print ('X', end='\t')
                
            print ('\n')
            #after each line we want to # print  a new line
        print ('------------------------------------------')
        
    

    def get_state(self):
        #this function returns the grid itsself and binary representation
        # 0 indicates that the cell is clear
        # 1 indicates that the cell is wall
        # 2 indicates the cell contains a marker
        # 
        map = np.zeros((2,self.m, self.n))
        for i,j in self.wallLocations:
            map[0][i][j] += 1
            map[1][i][j] += 1
            #if the locatin is wall then add 1
        for i,j in self.markers_locations:
            map[0][i][j] += 2
            #if marker then add 2 
        direction = ["north", "south", "east", "west"].index(self.orientation)
        direction = 2**(direction + 2)
        #using index from the orientation
        # north:4, south: 8, east: 16, west: 32
        
        i,j = self.getAgentRowAndColumn()
        # print  (self.agentPosition, i, j, self.orientation)
        map[0][i][j] += direction
        
        # final desired state, map[1]
        
        for i,j in self.terminalStates[2]:
            map[1][i][j] += 2
            
        final_dir = ["north", "south", "east", "west"].index(self.terminalStates[1])
        final_dir = 2**(final_dir + 2)
        i,j = self.terminalStates[0]
        map[1][i][j] += final_dir
        
        return map
        
            
            
    def actionSpaceSample(self):
        #return just a random choice of list of possible options in the system
        return np.random.choice(self.possibleActions)

