### Policy Gradient Actor Critic

- Simple Actor Critic  
	- [player_A2C_1](../reinforcement/players/player_A2C_1.py)
	- [player_A2C_1A](../reinforcement/players/player_A2C_1A.py)

### Overview

This Policy Gradients Actor Critic implementation is an Off-Policy Temporal Difference Actor Critic algorithm.  
It has an Experience Replay to sample experiences from different policies to calculate the TD Erros.  
It consists of having two networks, one that represents the policy (the Actor), and one that provides the state-values (the Critic).  

### References

SUTTON, Richard S; BARTO, Andrew G. Reinforcement Learning: An Introduction. The MIT Press, 2017.  
