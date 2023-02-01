import numpy as np
import gym
from reinforce_torch import PolicyGradientAgent
import matplotlib.pyplot as plt
from utils import plotLearning
from gym import wrappers
from env import GridWorld

if __name__ == '__main__':
    agent = PolicyGradientAgent(ALPHA=0.000001, input_dims=[32], GAMMA=0.99,
                                n_actions=6, layer1_size=128, layer2_size=128)
    #agent.load_checkpoint()
    # env = gym.make('LunarLander-v2')
    env = GridWorld(4, 4, (0, 1), 'east', [], [(0, 2), (1, 0), (1, 2), (1, 3), (2, 1), (2, 2), (3, 0), (3, 1), (3, 3)], [(1, 1), 'south', [(0, 1)]], ['m', 'l', 'r', 'f', 'pick', 'put'])
    score_history = []
    score = 0
    num_episodes = 5000
    #env = wrappers.Monitor(env, "tmp/lunar-lander",
    #                        video_callable=lambda episode_id: True, force=True)
    for i in range(num_episodes):
        # print('episode: ', i,'score: ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation.flatten())
            actionStrings =  ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']

            observation_, reward, done, info = env.step(actionStrings[action])
            agent.store_rewards(reward)
            observation = observation_
            score += reward
            if reward >= env.winReward:
                print("we solved the environment")
        score_history.append(score)
        agent.learn()
        #agent.save_checkpoint()
    filename = 'lunar-lander-alpha001-128x128fc-newG.png'
    plotLearning(score_history, filename=filename, window=25)
