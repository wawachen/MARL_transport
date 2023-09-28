import argparse
import torch
import os
import numpy as np
# from gym.spaces import Box, Discrete
from os.path import dirname, join, abspath
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from pyrep.baselines.buffer import ReplayBuffer
from pyrep.baselines.attention_sac import AttentionSAC

#################################
from pyrep.envs.bacterium_environment_mpi_RL_discrete import Drone_Env
from pyrep.common.arguments_MAAC import get_args
from gym import spaces
#################################


def run(config):
    ##########################################
    # get the params
    if config.field_size == 10:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
    if config.field_size == 15:
        env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')
    
    restart_frequency = 1000
    
    # create multiagent environment
    ##########################################

    model_dir = Path("./" + config.scenario_name + "/model_drone{}".format(config.n_agents)+'/'+'field_size{}'.format(config.field_size))

    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    # torch.manual_seed(run_num)
    # np.random.seed(run_num)
    env = Drone_Env(config,env_name,config.n_agents)
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, config.n_agents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, spaces.Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        if ((ep_i%restart_frequency)==0)and(ep_i!=0):
            env.restart()
        obs = torch.tensor(env.reset_world()).unsqueeze(0) #[n_agent,obs_size]->[env_n,n_agent,obs_size]
        print("Episodes %i of %i" % (ep_i + 1, config.n_episodes))
        # obs = env.reset()
        model.prep_rollouts(device='cpu')
        score = 0

        for et_i in range(config.episode_length):
            # print(np.vstack(obs[:, 0]).shape)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(config.n_agents)]
            # get actions as torch Variables
            # print(torch_obs[0].shape)
            torch_agent_actions = model.step(torch_obs, explore=True)
            # print(torch_agent_actions)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones = env.step(actions[0])
            score += rewards[0]
            # next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, torch.tensor(rewards).unsqueeze(0), torch.tensor(next_obs).unsqueeze(0), torch.tensor(dones).unsqueeze(0))
            obs = torch.tensor(next_obs).unsqueeze(0)
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
            if np.any(dones):
                break
        # ep_rews = replay_buffer.get_average_rewards(
        #     config.episode_length * config.n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        print("baseline{}_episode_e{}".format(config.n_agents,config.field_size),ep_i,":",score)
        logger.add_scalar('all_agents/mean_episode_rewards',
                              score, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    config = get_args()

    run(config)
