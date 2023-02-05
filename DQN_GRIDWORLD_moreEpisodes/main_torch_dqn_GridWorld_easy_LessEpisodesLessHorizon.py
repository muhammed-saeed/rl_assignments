# import gym
from simple_dqn_torch_mediumSize import Agent
from utils import plotLearning
import numpy as np
from env import GridWorld
from read_env_json import read_env_sol_json
from memory import get_memory
import torch as T

train_path = "/home/CE/musaeed/project/datasets/data_easy/train/task"
train_target_path = "/home/CE/musaeed/project/datasets/data_easy/train/seq"
if __name__ == '__main__':
    # env = gym.make('LunarLander-v2')
    env = GridWorld(4, 4, (0, 1), 'east', [], [(0, 2), (1, 0), (1, 2), (1, 3), (2, 1), (2, 2), (3, 0), (3, 1), (3, 3)], [(1, 1), 'south', [(0, 1)]], ['m', 'l', 'r', 'f', 'pick', 'put'])
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.05,
                  input_dims=[32], lr=0.00001)
    scores, eps_history = [], []
    tasks, optimumSolution, files = read_env_sol_json("train", train_path, train_target_path)
    unSolvedOPtimSeq = "/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/easy/unSolvedOptimimumSolutions/"
    unsolvedEnvs = "/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/easy/unSolvedEnvs/"
    n_games = 1200
    agent.Q_eval.load_state_dict(T.load("/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/easy/checkpoints/agent.pt"))
    for count, task in enumerate(tasks):
        isSolved = False
        action_seq = []
        counter = []
        env = GridWorld(*task)
        print(f"We are in task {count}")
        for i in range(n_games):
            if (i+1)%400 == 0:
                print(f"we are in game {i}")
            score = 0
            done = False
            observation = env.reset()
            eps_actions = []
            t = 0
            horizon = 50
            for t in range(horizon):
                while not done:
                    action = agent.choose_action(observation.flatten())
                    actions = ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']
                    # print(actions[action])
                    observation_, reward, done, info = env.step(actions[action])
                    score += reward
                    eps_actions.append(actions[action])
                    if reward >= env.winReward:
                        isSolved = True
                        # print("Solved")
                        # print(f"actions seq {eps_actions}")
                        action_seq.append(eps_actions)
                        counter.append(i)
                    
                    agent.store_transition(observation.flatten(), action, reward, 
                                            observation_.flatten(), done)
                    agent.learn()
                    observation = observation_
            scores.append(score)
            eps_history.append(agent.epsilon)

            avg_score = np.mean(scores[-100:])
        if isSolved:
            
            with open("/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/easy/solutions_seq/"+str(count)+".txt", "w") as fb:
                for number, solution in enumerate(action_seq):
                    fb.write(f"The solution occured at {counter[number]} episode \n")
                    fb.write(" ".join(solution))
                    fb.write('\n')
                    fb.write(f'Optimum Solution is {optimumSolution[count]}')
                    fb.write('\n-------------- -------------- ---------- -------- \n') 
        else:
            print(f"task {task} is unsloved and !")
            with open(unsolvedEnvs+str(count)+".txt", "w") as fb:
                fb.write(str(task))
            with open(unSolvedOPtimSeq+str(count)+".txt","w") as fb:
                fb.write(str(optimumSolution[count]))

            # print('episode ', i, 'score %.2f' % score,
                    # 'average score %.2f' % avg_score,
                    # 'epsilon %.2f' % agent.epsilon)
        T.save(agent.Q_eval.state_dict(),"/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/easy/checkpoints/agent.pt")
    x = [i+1 for i in range(n_games)]
    filename = '/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/easy/grid_world.png'
    plotLearning(x, scores, eps_history, filename)

