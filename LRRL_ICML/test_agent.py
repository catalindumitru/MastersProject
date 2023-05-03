import torch.cuda
from torch.utils.tensorboard.writer import SummaryWriter
import pickle
from parameters_experiments import *
from deep_rl import *
from deep_rl.utils.noise_generator import make_noise
import os

noise_list = {'MiniGrid-LavaGapS6-v0': {'bound': 2, 'var': 0.5},
              'MiniGrid-Dynamic-Obstacles-6x6-v0': {'bound': 2, 'var': 0.5},
              'MiniGrid-LavaCrossingS9N1-v0': {'bound': 1.5, 'var': 0.5}}

if __name__ == '__main__':
    con = Config()
    dir = 'data'
    models = []
    stats = []
    algos = []
    envs = []
    for file in os.listdir(dir):
        if '.model' in file:
            models.append(os.path.join(os.getcwd(),dir,file))
            algo = file[file.find('algo_')+5:file.find('algo_')+8]
            if algo == 'SAP':
                algo = 'SAPPO'
            elif algo == 'QA2':
                algo = 'QA2C'
            elif algo == 'DDP':
                algo = 'DDPG'
            algos.append(algo)
            env = file[file.find('-')+1:file.find('-algo')]
            envs.append(env)
        elif '.stats' in file:
            stats.append(os.path.join(os.getcwd(),dir,file))
    for model, stat, alg, e in zip(models,stats,algos,envs):
        agent = make_agent(algo=alg,game=e,robust=False)
        agent.load(model.replace('.model',''))
        noiseless = agent.eval_episodes()
        unif_noise = agent.eval_noisy_episodes(mode=1,bound=noise_list[e]['bound'])
        gaus_noise = agent.eval_noisy_episodes(mode=2,bound=noise_list[e]['bound'])
        adv_noise = [agent.eval_adv_episodes(0.5),agent.eval_adv_episodes(1),agent.eval_adv_episodes(2)]
        print("Results for model: ",model)
        print("Average rewards for noiseless environment: ", noiseless)
        print("Average rewards for uniform noise environment: ", unif_noise)
        print("Average rewards for gaussian noise environment: ", gaus_noise)
        print("Average rewards for adversarial noise environments: ", adv_noise)