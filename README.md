# Policy Gradient Methods on OpenAi LunarLander

This project implements the standard policy gradient algorithm (REINFORCE) and applies it to solve the lunar lander environment in OpenAi Gym

## Analysis of results

As can be seen in the results plot, the agent shows signs of learning and is able to solve the lunar lander environment (score of 200 points).

The first time where the agent solves the environment is at around episode 300. However, the learning is not very stable and the agent's performance deterorates after.

Early-stopping techniques can be implemented to save the best version of the agent while learning.

## Getting Started

1. Activate conda environment with dependencies installed
2. Run lunar_lander.py

### Prerequisites

Project requires: Pytorch v1.4.0 installed
Other dependencies include:
- os 
- Numpy
- gym
- Matplotlib

## Built With

* [numpy](https://numpy.org/) - Fundamental package for scientific computing with Python
* [Pytorch](https://pytorch.org/) - Deep learning Framework used along with Numpy to build Deep Q Networks.
* [OpenAI Gym](https://gym.openai.com/) - Provides environments to test Agent's performance

## Acknowledgments

This project was built referencing research papers on applying Q-learning with deep neural networks

**https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning**

**https://arxiv.org/abs/1509.06461**