### Proximal Policy Optimization

- PPO  
	- [player_PPO_1](../learning/players_reinforcement/player_PPO_1.py)
	- [player_PPO_1A](../learning/players_reinforcement/player_PPO_1A.py)
	- [player_PPO_2](../learning/players_reinforcement/player_PPO_2.py)
	- [player_PPO_2A](../learning/players_reinforcement/player_PPO_2A.py)

### Overview

The Proximal Policy Optimization is an On-Policy with Importance Sampling Actor Critic algorithm that alternates between colecting batches of samples and optimizing the policy by a number of epochs with a “surrogate” objective function.  
This implementation does not share parameters between the Actor and the Critic. It does not have parallel actors, so the sampled batch comes from a single agent.  
It works on continuous or discrete action space environments.  

### References

SCHULMAN, John; et al. Proximal Policy Optimization Algorithms. aug. 2017.  
SCHULMAN, John; et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. sep. 2017  
HEESS, Nicolas; et al. Emergence of Locomotion Behaviours in Rich Environments. jul. 2017  
SCHULMAN, John; et al. Trust region policy optimization. ICML. 2015.  
