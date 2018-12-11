### Generative Adversarial Imitation Learning

- GAIL  
	- [player_GAIL_1](../learning/players_imitation/player_GAIL_1.py)
	- [player_GAIL_1A](../learning/players_imitation/player_GAIL_1A.py)

### Overview

The Generative Adversarial Imitation Learning algorithm uses a set of demonstrated trajectories D = {s,a} to imitate the expert's policy. It is inspired by the Generative Adversarial Network algorithm and by Inverse Reinforcement Learning algorithms.  

The agent’s policy receives a state and acts in the environment, while a discriminator classifies the trajectories and provides a signal for the policy to learn. The goal is to generate a distribution that matches the expert distribution. In the originar paper the policy optimization step is done using TRPO, but in this implementation PPO is used.  

### References

[Ho J, Ermon S. Generative adversarial imitation learning. NIPS 2016](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf)  
