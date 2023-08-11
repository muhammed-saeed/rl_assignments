## Reinforcement Learning Environment for Dynamic Programming and Deep Reinforcement Learning

This repository contains Python code for an environment designed to facilitate the training and evaluation of reinforcement learning models, particularly for dynamic programming and deep reinforcement learning tasks. The environment is structured around an `Env` class, offering a comprehensive set of properties and functionalities for creating and simulating various scenarios.

## Features

- **Environment Initialization:** The environment can be initialized from scratch. It supports custom initial states, allowing you to set an initial state or choose a random one.

- **Goal-Oriented:** The environment includes a goal state that can be predefined or set dynamically. It implements a complex reward system. For instance, if the agent takes an action to move forward and its goal is to move forward, a positive reward associated with moving is given. The same applies to actions like turning right and turning left.

- **Boundary Check:** The `Env` class checks the agent's location against the boundaries of the environment before executing movement actions. This ensures that the agent remains within the defined bounds.

- **Board Dimensions:** The environment's dimensions are provided as input to the `Env` class. These dimensions are used to define the boundaries of the board, offering flexibility in setting up the environment.

- **Goal Achievement:** The environment assesses whether the agent has reached the goal state. Upon successful achievement, a positive "goal reward" is granted.

- **Interaction with Objects:** Actions such as PickUp and Put are supported. The environment verifies whether the agent's hand is empty before executing these actions, adjusting the state accordingly based on the selected action.

- **Planning Horizon:** Before each action, the environment evaluates the remaining planning horizon for the agent. This consideration influences the selection of further actions.

## Getting Started

1. Clone this repository to your local machine.

2. Ensure you have the necessary dependencies installed by running:

   ```bash
   pip install -r requirements.txt
