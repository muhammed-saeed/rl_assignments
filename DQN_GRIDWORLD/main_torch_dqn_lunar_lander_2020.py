import gym
from simple_dqn_torch_2020 import Agent
from utils import plotLearning
import numpy as np
from env import GridWorld
from read_env_json import read_env_sol_json

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env = GridWorld(4, 4, (0, 1), 'east', [], [(0, 2), (1, 0), (1, 2), (1, 3), (2, 1), (2, 2), (3, 0), (3, 1), (3, 3)], [(1, 1), 'south', [(0, 1)]], ['m', 'l', 'r', 'f', 'pick', 'put'])
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01,
                  input_dims=[32], lr=0.00001)
    scores, eps_history = [], []
    n_games = 2000
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation.flatten())
            actions = ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']
            # print(actions[action])
            observation_, reward, done, info = env.step(actions[action])
            score += reward
            if reward >= env.winReward:
                print("Solved")
                print(actions[action])
            agent.store_transition(observation.flatten(), action, reward, 
                                    observation_.flatten(), done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)

