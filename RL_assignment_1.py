class Env:
    def __init__(self, discount_rate,action = "init", is_alive=True,  init_state = [0,0],goal_state=[100,100], boundries = [], bulbs_location =[], moving_reward=1, picking_reward=1, goal_reward=100, hand="empty" ):
        self.action = action
        self.is_alive = is_alive
        self.gamma = discount_rate
        self.init_state = init_state
        self.boundries = boundries
        self.isHandEmpty = hand
        self.moving_reward = moving_reward
        self.picking_reward = picking_reward
        self.goal_reward = goal_reward

    def check_if_hit_wall(self):
        if self.state  in self.boundries:
            self.is_alive = False
            print("Game over start new game")
            return self.is_alive
    #the interaction between the agent and the environment 
    
    def step(self, action):
        match self.action:
            case "init":
                self.state = self.init_state
                if self.check_if_hit_wall() :
                    
                    return False
                else: 
                    return self.reward
            case "turnLeft":
                self.state[0] -= 1
                if self.check_if_hit_wall() :
                    
                    return False
                else: 
                    return self.reward
            case "turnRight": 
                self.state[0] += 1
                if self.check_if_hit_wall() :
                    
                    return False
                else: 
                    return self.reward
            case "move":
                self.state[1] += 1
                if self.check_if_hit_wall() :
                    
                    return False
                else: 
                    return self.reward
            case "pick":
                self.state[1] -= 1
                if self.check_if_hit_wall() :
                    
                    return False
                else: 
                    return self.reward
        




class Agent:
    def __init__(self,state, reward, actions = []):
        self.state = state
        self.reward = reward
        self.actions = actions
    
    def select_action(self, state):
        asdf
    

actions = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker", "finish"]
if __name__ == "__main__":
    env  = Env(0.9)
    print(f"{env.gamma})
