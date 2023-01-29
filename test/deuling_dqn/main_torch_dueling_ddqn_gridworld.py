import gym
import numpy as np
from dueling_ddqn_torch import Agent
from utils import plotLearning
from read_env_json import read_env_sol_json
from env import GridWorld

mode = "train"
train_path = "/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/task/task"
train_target_path = "/home/muhammed-saeed/Documents/rl_assignments/Reinforce_policy_gradient_gridworld/task/solution"
# m, n, init_state, orientation, markers_locations, wall_locations, terminal_state, possible_actions):
# terminal_state #[[x,y],"orientation", [[markers1],[marker2]]]
actions = ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']
ENVS,SEQ,FILES = read_env_sol_json(mode, train_path, train_target_path)
print(f"{ENVS[0]} \n\n") # best action seq {initial_settings[1]}")
# print("#############################")
# a = initial_settings[0]
# print(len(a))
# print(initial_settings[0])
env = GridWorld(*ENVS[0])
if __name__ == '__main__':
    # env = gym.make('LunarLander-v2')
    num_games = 250
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4,
                  input_dims=[1,32], n_actions=6, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-3, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'LunarLander-Dueling-DDQN-512-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        if i > 0 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)
