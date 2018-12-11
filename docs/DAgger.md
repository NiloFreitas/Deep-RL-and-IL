### Dataset Aggregation

- DAgger  
	- [player_DAgger_1](../learning/players_imitation/player_DAgger_1.py)
	- [player_DAgger_1A](../learning/players_imitation/player_DAgger_1A.py)

### Overview

The Dataset Aggregation algorithm uses a set of demonstrated trajectories D = {s,a} to imitate the expert's policy. First, it trains a policy that mimics the expert in a supervised manner and then trains a policy using the aggregated dataset D = D + Di where Di = {s,pi(s)}.  

### References

[ROSS S, GORDON G, BAGNELL D. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. 2011](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf)  
