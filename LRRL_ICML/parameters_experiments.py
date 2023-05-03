import torch

from deep_rl import *
from deep_rl.utils.noise_generator import make_noise
env_list = {'mujoco':['Pendulum-v0','Hopper-v2','Ant-v3','Walker2d-v2'],
            'gridsafety': ['MiniGrid-DistShift2-v0','MiniGrid-DistShift1-v0',
                           'MiniGrid-LavaGapS5-v0','MiniGrid-LavaGapS6-v0','MiniGrid-LavaGapS7-v0',
                           'MiniGrid-LavaCrossingS9N1-v0','MiniGrid-LavaCrossingS9N2-v0',
                           'MiniGrid-LavaCrossingS9N3-v0','MiniGrid-LavaCrossingS11N5-v0',
                           'MiniGrid-Dynamic-Obstacles-5x5-v0','MiniGrid-Dynamic-Obstacles-Random-5x5-v0',
                           'MiniGrid-Dynamic-Obstacles-6x6-v0','MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
                           'MiniGrid-Dynamic-Obstacles-8x8-v0','MiniGrid-Dynamic-Obstacles-16x16-v0']}

def make_agent(**kwargs):
    generate_tag(kwargs)
    game = kwargs['game']
    algo = kwargs['algo']
    robust = kwargs['robust']
    if robust:
        noise = kwargs['noise']
        bound = kwargs['bound']
        var = kwargs['var']
    else:
        noise = None
        bound = None
        var = None
    pre_set_parameters = False
    kwargs.setdefault('log_level', 0)
    for key in env_list.keys():
        if game in env_list[key]:
            pre_set_parameters = True
            config = load_config(key,algo,game,robust,noise,bound=bound,var=var)
    if not pre_set_parameters: #load
        raise "Configuration not found."
    noisefun = config.noise
    config.merge(kwargs)
    if algo == 'PPO':
        return PPOAgent(config,noise=noisefun)
    elif algo == 'A2C':
        return A2CAgent(config,noise=noisefun)#A2CAgent(config,noise=noisefun)
    elif algo == 'QA2C':
        return QA2CAgent(config,noise=noisefun)
    elif algo == 'SAPPO':
        return SAPPOAgent(config,noise=noisefun)
    else:
        return DDPGAgent(config,noise=noisefun)


def load_config(tag,algo,game,robust,noise,bound=None,var=None):
    if tag == 'mujoco':
        return mujoco_con(algo,game,robust,noise,bound=bound,var=var)
    elif tag=='gridsafety':
        return gridsafety_con(algo,game,robust,noise,bound=bound,var=var)
    else:
        print("Environment for"+str(game)+ 'not found')

def mujoco_con(algo,game,robust=False,noise=None,bound=None,var=None):
    con = Config()
    if torch.cuda.is_available():
        con.DEVICE = torch.device('cuda:%d' % (0))
    else:
        con.DEVICE = torch.device('cpu')
    if robust:
        con.lexico = True
        con.noise_mode = noise
        con.noise = make_noise(game, variance=var,mode=noise, bound=bound)
    else:
        con.lexico = False
        con.noise = None
    if algo=='PPO':
        con.clip_rewards = False
        con.task_fn = lambda: Task(game,clip_rewards=con.clip_rewards,wrapper='mujoco')
        con.eval_env = Task(game,wrapper=None)
        con.actor_opt_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
        con.critic_opt_fn = lambda params: torch.optim.Adam(params, 1.5e-4, eps=1e-5)
        con.network_fn = lambda: GaussianActorCriticNet(
            con.state_dim, con.action_dim, actor_body=FCBody(con.state_dim,(64,64), gate=torch.tanh),
            critic_body=FCBody(con.state_dim,(64,64), gate=torch.tanh))
        con.state_normalizer = MeanStdNormalizer()
        con.reward_normalizer = RescaleNormalizer()
        con.discount = 0.99
        con.use_gae = True
        con.gae_tau = 0.95
        con.rollout_length = 2048
        con.entropy_weight = 0
        con.value_loss_weight = 0.1
        con.optimization_epochs = 10
        con.mini_batch_size = 32
        con.ppo_ratio_clip = 0.2
        con.log_interval = 1000
        con.eval_interval = 10000
        con.max_steps = int(2e6)
        con.target_kl = 0.01
        con.decaying_lr = True

    elif algo == 'DDPG':
        con.clip_rewards = False
        con.task_fn = lambda: Task(game, clip_rewards=con.clip_rewards,wrapper='mujoco')
        con.eval_env = Task(game)
        con.max_steps = int(2e6)
        con.eval_episodes = 20
        con.network_fn = lambda: DeterministicActorCriticNet(
            con.state_dim, con.action_dim,
            actor_body=FCBody(con.state_dim, (400, 300), gate=F.relu),
            critic_body=FCBody(con.state_dim + con.action_dim, (400, 300), gate=F.relu),
            actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
            critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4))
        con.state_normalizer = MeanStdNormalizer()
        con.replay_fn = lambda: UniformReplay(memory_size=int(1000000), batch_size=64)
        con.discount = 0.99
        con.eval_interval = 10000
        con.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
            size=(con.action_dim,), std=LinearSchedule(0.2))
        con.warm_up = int(1e4)
        con.target_network_mix = 1e-3
    return con


def gridsafety_con(algo,game,robust=False,noise=None,bound=2,var=0.5):
    con = Config()
    if torch.cuda.is_available():
        con.DEVICE = torch.device('cuda:%d' % (0))
    else:
        con.DEVICE = torch.device('cpu')
    if robust:
        con.lexico = True
        con.value_loss_weight = 1
        con.noise_mode = noise
        con.noise = make_noise(game, variance=var,mode=noise,bound=bound)
    else:
        con.lexico = False
        con.noise = None

    if algo == 'PPO':
        con.num_workers = 8
        con.clip_rewards = False
        con.task_fn = lambda: Task(game, num_envs=con.num_workers)
        con.eval_env = Task(game)
        con.network_fn = lambda: CategoricalActorCriticNet(con.state_dim, con.action_dim, GridConvBody())
        con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
        con.discount = 0.99
        con.use_gae = True
        con.gae_tau = 0.95
        con.gradient_clip = 0.5
        con.rollout_length = 128
        con.optimization_epochs = 10
        con.mini_batch_size = con.rollout_length * con.num_workers // 4
        con.ppo_ratio_clip = 0.2
        con.log_interval = 10000
        con.max_steps = int(6e5)
        con.shared_repr = True
        con.target_kl = 0.01
        con.state_normalizer = RescaleNormalizer()
        con.reward_normalizer = RescaleNormalizer()
        con.entropy_weight = 0.01
        if 'Gap' in game:
            # Specific parameters for LavaGap
            con.max_steps = int(1e6)
            con.discount = 0.99
            con.rollout_length = 256
            con.gradient_clip = 0.5#0.5
            con.entropy_weight = 0
            con.reward_normalizer = RescaleNormalizer()
        if 'Cross' in game:
            con.entropy_weight = 0#.0001
            con.task_fn = lambda: Task(game, num_envs=con.num_workers, a2cwrapper=2)
            con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
            con.discount = 0.99
            con.rollout_length = int(512)
            con.max_steps = int(1e7)
            con.mini_batch_size = con.rollout_length * con.num_workers // 4
            con.gradient_clip = 0.5
        if 'Dist' in game:
            con.reward_normalizer = RescaleNormalizer(10)
        if 'Dyn' in game:
            con.task_fn = lambda: Task(game, num_envs=con.num_workers, a2cwrapper=2)
            con.discount = 0.99
            con.rollout_length = int(256)
            con.entropy_weight = 0
            con.max_steps = int(6e5)
            con.mini_batch_size = con.rollout_length * con.num_workers // 4
            con.gradient_clip = 0.5
        # con.decaying_lr = True

    if algo == 'SAPPO':
        con.DEVICE = torch.device('cpu')
        con.num_workers = 8
        con.clip_rewards = False
        con.task_fn = lambda: Task(game, num_envs=con.num_workers)
        con.eval_env = Task(game)
        con.network_fn = lambda: CategoricalActorCriticNet(con.state_dim, con.action_dim, GridConvBody())
        con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
        con.discount = 0.99
        con.use_gae = True
        con.gae_tau = 0.95
        con.gradient_clip = 0.5
        con.rollout_length = 128
        con.optimization_epochs = 10
        con.mini_batch_size = con.rollout_length * con.num_workers // 4
        con.ppo_ratio_clip = 0.2
        con.log_interval = 10000
        con.max_steps = int(6e5)
        con.shared_repr = True
        con.target_kl = 0.01
        con.kppo = 1
        con.state_normalizer = RescaleNormalizer()
        con.reward_normalizer = RescaleNormalizer()
        con.entropy_weight = 0.01
        if 'Gap' in game:
            # Specific parameters for LavaGap
            con.max_steps = int(1e6)
            con.discount = 0.99
            con.rollout_length = 256
            con.gradient_clip = 0.5  # 0.5
            con.entropy_weight = 0
            con.reward_normalizer = RescaleNormalizer()
        if 'Cross' in game:
            con.entropy_weight = 0  # .0001
            con.task_fn = lambda: Task(game, num_envs=con.num_workers, a2cwrapper=2)
            con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
            con.discount = 0.99
            con.rollout_length = int(512)
            con.max_steps = int(1e7)
            con.mini_batch_size = con.rollout_length * con.num_workers // 4
            con.gradient_clip = 0.5
        if 'Dist' in game:
            con.reward_normalizer = RescaleNormalizer(10)
        if 'Dyn' in game:
            con.task_fn = lambda: Task(game, num_envs=con.num_workers, a2cwrapper=2)
            con.discount = 0.99
            con.rollout_length = int(256)
            con.entropy_weight = 0
            con.max_steps = int(6e5)
            con.mini_batch_size = con.rollout_length * con.num_workers // 4
            con.gradient_clip = 0.5

    elif algo == 'A2C':
        con.num_workers = 8
        con.task_fn = lambda: Task(game, num_envs=con.num_workers)
        con.eval_env = Task(game, num_envs=1)
        con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
        con.network_fn = lambda: CategoricalActorCriticNet(con.state_dim, con.action_dim, GridConvBody())
        con.state_normalizer = RescaleNormalizer()
        con.reward_normalizer = RescaleNormalizer()  # SignNormalizer()
        con.discount = 0.99
        con.use_gae = True
        con.gae_tau = 0.95
        con.entropy_weight = 0#0.01
        con.log_interval = 10000
        con.eval_interval = 10000
        con.rollout_length = 128
        con.gradient_clip = 0.5
        con.max_steps = int(6e5)
        if 'Gap' in game:
            # Specific parameters for LavaGap
            con.max_steps = int(1e6)
            con.discount = 0.99
            con.rollout_length = 256
            con.gradient_clip = 0.5
        if 'Cross' in game:
            con.task_fn = lambda: Task(game,single_process=False, num_envs=con.num_workers, a2cwrapper=2)
            con.eval_env = Task(game)
            con.discount = 0.999
            con.rollout_length = int(512)
            con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
            con.max_steps = int(8000000)
            con.gradient_clip = 0.5
            con.entropy_weight = 0#0.001
            con.reward_normalizer = RescaleNormalizer(10)
        if 'Dist' in game:
            con.reward_normalizer = RescaleNormalizer(10)
        if 'Dyn' in game:
            con.task_fn = lambda: Task(game, single_process=False, num_envs=con.num_workers, a2cwrapper=2)
            con.eval_env = Task(game)
            con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
            con.discount = 0.99
            con.rollout_length = int(256)
            con.max_steps = int(8e5)
            con.mini_batch_size = con.rollout_length * con.num_workers // 4
            con.gradient_clip = 0.5
    elif algo == 'QA2C':
        con.num_workers = 8
        con.task_fn = lambda: Task(game, num_envs=con.num_workers)
        con.eval_env = Task(game)
        con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
        con.network_fn = lambda: CategoricalQActorCriticNet(con.state_dim, con.action_dim, GridConvBody())
        con.state_normalizer = RescaleNormalizer()
        con.reward_normalizer = RescaleNormalizer()  # SignNormalizer()
        con.replay_fn = lambda: UniformReplay(memory_size=int(1e5), batch_size=32)
        con.warm_up = 1e4
        con.discount = 0.99
        con.use_gae = True
        con.eval_interval = 1000000
        con.gae_tau = 0.95
        con.entropy_weight = 0#0.01
        con.log_interval = 1000000
        con.rollout_length = 128
        con.gradient_clip = 0.5
        con.max_steps = int(6e5)
        if 'Gap' in game:
            # Specific parameters for LavaGap
            con.task_fn = lambda: Task(game, num_envs=con.num_workers, a2cwrapper=1)
            con.max_steps = int(2e6)
            con.discount = 0.99
            con.rollout_length = 256
            con.gradient_clip = 0.5
            con.entropy_weight = 0.001
            con.noise = make_noise(game, variance=1, mode=noise, bound=2)
            con.reward_normalizer = RescaleNormalizer(10)
        if 'Cross' in game:
            con.task_fn = lambda: Task(game,single_process=False, num_envs=con.num_workers, a2cwrapper=2)
            con.eval_env = Task(game)
            con.discount = 0.99
            con.rollout_length = int(512)
            con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
            con.max_steps = int(8000000)
            con.gradient_clip = 0.5
            con.entropy_weight = 0#.001
            con.reward_normalizer = RescaleNormalizer(10)
        if 'Dist' in game:
            con.reward_normalizer = RescaleNormalizer(10)
        if 'Dyn' in game:
            con.task_fn = lambda: Task(game, num_envs=con.num_workers, a2cwrapper=2)
            con.eval_env = Task(game)
            con.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.001, eps=1e-8)
            con.discount = 0.99
            con.rollout_length = int(256)
            con.max_steps = int(1e6)
            con.mini_batch_size = con.rollout_length * con.num_workers // 4
            con.gradient_clip = 0.5
    return con