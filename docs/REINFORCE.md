### REINFORCE

- REINFORCE  

	- [player_reinforce_1](../learning/players_reinforcement/player_reinforce_1.py) and [player_reinforce_1A](../learning/players_reinforcement/player_reinforce_1A.py)  

	- [player_reinforce_2](../learning/players_reinforcement/player_reinforce_2.py) and [player_reinforce_2A](../learning/players_reinforcement/player_reinforce_2A.py)  

	- [player_reinforce_rnn_1](../learning/players_reinforcement/player_reinforce_rnn_1.py) and [player_reinforce_rnn_1A](../learning/players_reinforcement/player_reinforce_rnn_1A.py)  

	- [player_reinforce_rnn](../learning/players_reinforcement/player_reinforce_rnn_2.py) and [player_reinforce_rnn_2A](../learning/players_reinforcement/player_reinforce_rnn_2A.py)  

*1A* features only MLP;  *2A* features Conv. Nets.;  *rnn* features LSTM Rec. Nets.  

### Overview

REINFORCE is an On-Policy Monte Carlo Policy-Based algorithm that learns an stochastic policy using Policy Gradients.  
It uses the discounted Monte Carlo returns and the log likelihood ratio to optimize the policy directly.  

### References

WILLIAMS, Ronald. Simple statistical gradient-following algorithms for connectionist reinforcement learning. 1992.  
