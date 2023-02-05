import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
from read_env_json import read_env_sol_json
from memory import get_memory
from env import GridWorld
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 1e-5


    train_path = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_easy/train/task"
    train_target_path = "/home/muhammed-saeed/Documents/rl_assignments/project/datasets/data_easy/train/seq"


    scores, eps_history = [], []
    tasks, optimumSolution, files = read_env_sol_json("train", train_path, train_target_path)
    unSolvedOPtimSeq = "/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/mediumSize/unSolvedOptimimumSolutions/"
    unsolvedEnvs = "/home/CE/musaeed/DQN_GRIDWORLD_moreEpisodes/mediumSize/unSolvedEnvs/"
    agent = Agent(n_actions=6, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=(32,))
    n_games = 4000

    figure_file = '/home/muhammed-saeed/Documents/rl_assignments/PPO/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    actions = []
    actions = ['move', 'left', 'right', 'finish', 'pickMarker', 'putMarker']
    env = GridWorld(4, 4, (0, 1), 'east', [], [(0, 2), (1, 0), (1, 2), (1, 3), (2, 1), (2, 2), (3, 0), (3, 1), (3, 3)], [(1, 1), 'south', [(0, 1)]], ['m', 'l', 'r', 'f', 'pick', 'put'])
   

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        eps_actions=  []
        while not done:
            action, prob, val = agent.choose_action(observation.flatten())
            actionString = actions[action]
            # print(f"ACtions is {actionString}")
            observation_, reward, done, info = env.step(actionString)
            n_steps += 1
            score += reward
            eps_actions.append(actionString)
            if reward >= env.winReward:
                print(f"we solved it ")
                print(f"and sequence is {eps_actions}")
            agent.remember(observation.flatten(), action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters , " last action ", eps_actions[-1])
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


