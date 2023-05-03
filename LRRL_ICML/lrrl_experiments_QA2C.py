import torch.cuda
from torch.utils.tensorboard.writer import SummaryWriter
import pickle
from parameters_experiments import *
from deep_rl import *
from deep_rl.utils.noise_generator import make_noise

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    seed = np.random.randint(int(1e6))
    random_seed(seed)
    select_device(-1)
    results = {}
    iterations = 10

    game_list = ['MiniGrid-LavaCrossingS9N1-v0']

    for game in game_list:
        results[game] = {'PPO_vanilla': {'noise':[],'no_noise':[],'final':{}},
                         'PPO_LRRL_noise0': {'noise':[],'no_noise':[],'final':{}},
                         'PPO_LRRL_noise2': {'noise':[],'no_noise':[],'final':{}},
                         'A2C_vanilla': {'noise':[],'no_noise':[],'final':{}},
                         'A2C_LRRL_noise0': {'noise':[],'no_noise':[],'final':{}},
                         'A2C_LRRL_noise2': {'noise':[],'no_noise':[],'final':{}},
                         'QA2C_vanilla': {'noise': [], 'no_noise': [], 'final': {}},
                         'QA2C_LRRL_noise0': {'noise': [], 'no_noise': [], 'final': {}},
                         'QA2C_LRRL_noise2': {'noise': [], 'no_noise': [], 'final': {}},
                         'DDPG_vanilla': {'noise':[],'no_noise':[],'final':{}},
                         'DDPG_LRRL_noise0': {'noise':[],'no_noise':[],'final':{}},
                         'DDPG_LRRL_noise2': {'noise':[],'no_noise':[],'final':{}},
                         'seed':seed}
        noise_list = {'MiniGrid-LavaGapS6-v0': {'bound': 2, 'var': 0.5},
                      'MiniGrid-Dynamic-Obstacles-6x6-v0': {'bound': 2, 'var': 0.5},
                      'MiniGrid-LavaCrossingS9N1-v0': {'bound': 1.5, 'var': 0.5}}
          # no noise
        for _ in range(iterations):
            # QA2C Agents
            agent = make_agent(game=game,algo='QA2C',robust=True,noise=1,bound=noise_list[game]['bound'],var=noise_list[game]['var'])
            run_steps(agent)
            results[game]['QA2C_LRRL_noise0']['no_noise'].append(agent.eval_episodes()['episodic_return_test'])
            results[game]['QA2C_LRRL_noise0']['noise'].append([agent.eval_noisy_episodes(mode=1,bound=noise_list[game]['bound'])['episodic_return_noise'],
                                                          agent.eval_noisy_episodes(mode=2,bound=noise_list[game]['bound'])['episodic_return_noise'],
                                                               agent.eval_adv_episodes(0.5),agent.eval_adv_episodes(1),agent.eval_adv_episodes(2)])
            agent = make_agent(game=game,algo='QA2C',robust=True,noise=2,bound=noise_list[game]['bound'],var=noise_list[game]['var'])
            run_steps(agent)
            results[game]['QA2C_LRRL_noise2']['no_noise'].append(agent.eval_episodes()['episodic_return_test'])
            results[game]['QA2C_LRRL_noise2']['noise'].append([agent.eval_noisy_episodes(mode=1,bound=noise_list[game]['bound'])['episodic_return_noise'],
                                                          agent.eval_noisy_episodes(mode=2,bound=noise_list[game]['bound'])['episodic_return_noise'],
                                                               agent.eval_adv_episodes(0.5),agent.eval_adv_episodes(1),agent.eval_adv_episodes(2)])

            for key in results[game].keys():
                try:
                    zero_noise_median = np.median(results[game][key]['no_noise'])
                    zero_noise_std = np.std(results[game][key]['no_noise'])
                    unif_noise_median = np.median([i[0] for i in results[game][key]['noise']])
                    unif_noise_std = np.std([i[0] for i in results[game][key]['noise']])
                    gaus_noise_median = np.median([i[1] for i in results[game][key]['noise']])
                    gaus_noise_std = np.std([i[1] for i in results[game][key]['noise']])
                    noise_median_05 = np.median([i[2] for i in results[game][key]['noise']])
                    noise_std_05 = np.std([i[2] for i in results[game][key]['noise']])
                    noise_median_1 = np.median([i[3] for i in results[game][key]['noise']])
                    noise_std_1 = np.std([i[3] for i in results[game][key]['noise']])
                    noise_median_2 = np.median([i[4] for i in results[game][key]['noise']])
                    noise_std_2 = np.std([i[4] for i in results[game][key]['noise']])
                    results[game][key]['final'] = {'zero_noise_median': zero_noise_median,
                                                   'zero_noise_std': zero_noise_std,
                                                   'unif_noise_median': unif_noise_median,
                                                   'unif_noise_std': unif_noise_std,
                                                   'gaus_noise_median': gaus_noise_median,
                                                   'gaus_noise_std': gaus_noise_std,
                                                   'adv_noise_median': [noise_median_05, noise_median_1,
                                                                        noise_median_2],
                                                   'adv_noise_std': [noise_std_05, noise_std_1, noise_std_2]}
                except:
                    None
            with open('results_FINAL_discrete_QA2C_'+game+'.pickle','wb') as f:
                pickle.dump(results,f)
