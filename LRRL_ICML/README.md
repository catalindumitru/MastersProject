# LRPG
Lexicographically-Robust Policy Gradient Algorithms

Main algorithm implementation is based on https://github.com/ShangtongZhang/DeepRL, adding Lexicographic Robustness for DDPG, PPO and A2C.

The files "lrrl_experiments_A2C.py", "lrrl_experiments_PPO.py", "lrrl_experiments_SAPPO.py" and "lrrl_experiments_QA2C.py" run the experiments for the 
Minigrid environments presented in Table 1 and save the results to a pickle file. All the configuration parameters are pre-loaded in 'parameter_experiments.py'. 
The package also supports DDPG implementations, but it is not used for the experiments in the paper. 

The scripts test agents against adversarial noise as described in the Appendix.

The learned agents are saved in the folder './data'. To test all the saved agents, simply run:
python test_agents.py

A general implementation of the given algorithms for any environment is forthcoming.

## Installing Conda Environment

We provide a yml file containing the pip dependencies needed to run the experiment scripts. With Anaconda installed, run

conda env create -f lrrl.yml

