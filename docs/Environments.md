## Enviroments

The suppported Environments are in the [sources](../learning/sources) folder.  
The [source.py](../reinforcement/sources/source.py) contains the parent class.  
Currently, there are interfaces with [OpenAI's Gym](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/blob/master/reinforcement/sources/source_gym.py), [PyGame](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/blob/master/reinforcement/sources/source_pygame.py), [Unity ML](https://github.com/NiloFreitas/Deep-Reinforcement-Learning/blob/master/reinforcement/sources/source_unity.py) and [CARLA](https://github.com/carla-simulator/carla).  

The Environments in the folder are:

Vector Inputs (1 dimension):  

	- Pygame
		- Chase
	- Gym
		- CartPole  
		- Continuous Mountain Car
		- Mujoco
			- Pendulum
			- HalfCheetah
			- Hopper
			- Reacher
	- Unity Machine Learning
		- 3D Ball

Image Inputs (2 dimensions):  

	- Pygame
		- Catch
	- Gym
		- Breakout
		- Pong  
	- CARLA

## Datasets

	The datasets currently stored are:

		- Gym Cartpole
		- Unity 3DBall
		- Gym HalfCheetah

### Notes

1) It is easy to add new environments.

3) There is a need to place the unity build files of the respective environments in the *learning/sources/unity/* folder

2) To export a Unity Env .bytes file to run the trained model on Unity:  
Execute this function on the algorithm when a saved trained model of the env is in _trained_models_ folder:     
	<sub>from sources.source_unity_exporter import *  
	export_ugraph (self.brain, "./trained_models/" + trainedmodel, envname, nnoutput)  
	raise SystemExit(0)  
	#Example with PPO and 3DBall: trainedmodel = "unity_3dball_ppo/",
	                              envname = "3dball",
				      nnoutput =  "Actor/Mu/MatMul" </sub>  

3)  To use [CARLA](https://github.com/carla-simulator/carla) download the compiled version and put the files in the _/sources/carla/_ folder  
    Credits to the code in _sources/carla/Environment/_:  [Gokul](https://github.com/GokulNC/Setting-Up-CARLA-RL)
