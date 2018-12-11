### Deep Deterministic Policy Gradients

- DDPG  
	- [player_DDPG_1](../learning/players_reinforcement/player_DDPG_1.py)
	- [player_DDPG_1A](../learning/players_reinforcement/player_DDPG_1A.py)

### Overview

The Deep Deterministic Policy Gradients is an Off-Policy Temporal Difference Actor-Critic algorithm.  
It has an Experience Replay to sample experiences from different policies to calculate the TD Erros.  
It uses Target Networks, updated softly, to stabilize the action-value functions estimations.  
It operates on continuous action spaces and uses the Uhlenbeck and Ornstein process to improve exploration.  

### References

LILLICRAP, Timothy P.; et al. Continuous control with deep reinforcement learning. ICLR, fev. 2016.  
SILVER, David; et al. Deterministic policy gradient algorithms. ICML, 2014.  
