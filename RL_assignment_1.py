import numpy as np
import random
class Env:
    def __init__(self,  discount_rate,horizon = 100,action = "init", is_alive=True,  init_state = (0,0),goal_state=(100,100), board_dim = [28,28], markers_locations =[], moving_reward=1, picking_reward=1, false_pick_reward = 0,  hit_wall_punish = -100,goal_reward=100, isHandEmpty=True ):
        
        self.horizon = horizon
        self.action = action
        self.is_alive = is_alive
        self.gamma = discount_rate
        self.init_state = init_state
        # self.init_state = random.sample(range(10, 30), 2)
        self.board_dim = board_dim
        self.boundries = [[i,0] for i in range(self.board_dim[0])]
        self.boundries.extend([[0,i] for i in range(self.board_dim[1])])
        self.boundries.extend([[i,self.board_dim[1]] for i in range(self.board_dim[0])])
        self.boundries.extend([[self.board_dim[0], i] for i in range(self.board_dim[1])])




        self.isHandEmpty = isHandEmpty

        self.goal_state = goal_state
        self.reward = 0
        self.hit_wall_punish = hit_wall_punish
        self.moving_reward = moving_reward
        self.picking_reward = picking_reward
        self.goal_reward = goal_reward

        self.goal_state = goal_state
        self.markers_locations = markers_locations
        self.false_pick_reward = false_pick_reward
        
        
    def check_if_hit_wall(self):
        if self.state  in self.boundries:
            self.is_alive = False
            print("Game over start new game")
            return self.is_alive
    #the interaction between the agent and the environment 
    
    def step(self, action):
        if self.alive:
                match self.action:
                    case "init":
                        pass
                    case "turnLeft":
                        
                        if self.horizon < 0:
                            self.reward = -10000
                            print("The timesteps are ended")
                            return
                        
                        if self.check_if_hit_wall() :
                            self.reward  += self.hit_wall_punish
                            self.alive = False
                            return self.reward
                        else:
                            self.state[0] -= 1
                            #turn left reduce in the x-axis
                            self.horizon -= 1
                            self.reward += self.moving_reward 
                            return self.reward
                    case "turnRight": 
                        
                        if self.horizon < 0:
                            self.reward = -10000
                            print("The timesteps are ended")
                            return
                        if self.check_if_hit_wall() :
                            self.reward  += self.hit_wall_punish
                            self.alive = False
                            return self.reward
                        else:
                            self.state[0] += 1
                            #turn right increase in the x-axis

                            self.horizon -= 1
                            self.reward += self.moving_reward 
                            return self.reward
                    case "move":
                        
                        if self.horizon < 0:
                            self.reward = -10000
                            print("The timesteps are ended")
                            return
                        self.state[1] += 1
                        if self.check_if_hit_wall() :
                            self.reward  += self.hit_wall_punish
                            self.alive = False
                            return self.reward
                        else:
                            self.state[1] += 1
                            #move increase in the y axis
                            self.horizon -= 1
                            self.reward += self.moving_reward 
                            return self.reward

                    case "pickMarker":
                        
                        if self.horizon < 0:
                            self.reward = -10000
                            print("The timesteps are ended")
                            return
                        if self.isHandEmpty:
                            if self.state in self.markers_locations :
                                self.horizon -= 1 
                                self.isHandEmpty = False
                                self.reward += self.picking_reward
                        else:
                            
                            self.reward  += self.false_pick_reward
                            return self.reward


                    case "putMarker":
                        
                        if self.horizon < 0:
                            self.reward = -10000
                            print("The timesteps are ended")
                            return
                        if self.isHandEmpty:
                            self.reward += self.false_pick_reward
                        else:
                            self.horizon -= 1
                            if self.state in self.markers_locations:
                                self.reward += self.put_reward
                            else:
                                self.markers_locations.append(self.state)
                                self.reward += self.put_reward


                    case finish:
                        if self.horizon >=0 :
                            if self.state == self.goal_state:
                                self.reward += self.goal_reward
                            else:
                                pass

                        
        else:
            print("The agent died")
            return    

        




class Agent:
    def __init__(self,state, reward, actions = []):
        self.state = state
        self.reward = reward
        self.actions = actions
    
    def select_action(self, state):
        #here should have our brain Maybe using Q-Learning
        pass
    

actions = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]
if __name__ == "__main__":
    env  = Env(0.9)
    print(f"{env.gamma} \t {env.reward}")
