#!/usr/bin/env python
# Created at 2020/3/10
import math
import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pyrep.policies.GAE import estimate_advantages
from pyrep.policies.JointPolicy import JointPolicy
from pyrep.policies.ppo_step import ppo_step
from pyrep.common.ExpertDataSet import ExpertDataSet
from pyrep.networks.mlp_critic import Value
from pyrep.networks.mlp_discriminator import Discriminator
from pyrep.common.torch_util import device, to_device

trans_shape_func = lambda x: x.reshape(x.shape[0] * x.shape[1], -1)


class MAGAIL:
    def __init__(self, config, logger):
        self.config = config
        self.writer = logger

        self._load_expert_data()
        self._init_model()

    def _init_model(self):
        self.V = Value(num_states=self.config.value_num_states,
                       num_hiddens=self.config.value_num_hiddens,
                       drop_rate=self.config.value_drop_rate,
                       activation=self.config.value_activation)
        self.P = JointPolicy(self.expert_dataset.state.to(device),
                             self.config)
        self.D = Discriminator(num_states=self.config.discriminator_num_states,
                               num_actions=self.config.discriminator_num_actions,
                               num_hiddens=self.config.discriminator_num_hiddens,
                               drop_rate=self.config.discriminator_drop_rate,
                               use_noise=self.config.discriminator_use_noise,
                               noise_std=self.config.discriminator_noise_std,
                               activation=self.config.discriminator_activation)

        print("Model Structure")
        print(self.P)
        print(self.V)
        print(self.D)
        print()

        self.optimizer_policy = optim.Adam(self.P.parameters(), lr=self.config.jointpolicy_learning_rate)
        self.optimizer_value = optim.Adam(self.V.parameters(), lr=self.config.value_learning_rate)
        self.optimizer_discriminator = optim.Adam(self.D.parameters(), lr=self.config.discriminator_learning_rate)
        self.scheduler_discriminator = optim.lr_scheduler.StepLR(self.optimizer_discriminator,
                                                                 step_size=2000,
                                                                 gamma=0.95)

        self.discriminator_func = nn.BCELoss()

        to_device(self.V, self.P, self.D, self.D, self.discriminator_func)

    def _load_expert_data(self):
        expert_batch_size = self.config.general_expert_batch_size

        self.expert_dataset = ExpertDataSet(self.config.general_expert_data_path,
                                            self.config.n_agents)
        self.expert_data_loader = DataLoader(dataset=self.expert_dataset,
                                             batch_size=expert_batch_size,
                                             shuffle=True,
                                             num_workers=multiprocessing.cpu_count() // 2)

    def train(self, epoch, gen_batch):
        self.P.train()
        self.D.train()
        self.V.train()

        # collect generated batch
        # gen_batch = self.P.collect_samples(self.config["ppo"]["sample_batch_size"])
        # batch: ('state', 'action', 'next_state', 'log_prob', 'mask')
        gen_batch_state = trans_shape_func(
            torch.stack(gen_batch.state))  # [trajectory length * parallel size, state size]
        gen_batch_action = trans_shape_func(
            torch.stack(gen_batch.action))  # [trajectory length * parallel size, action size]
        gen_batch_next_state = trans_shape_func(
            torch.stack(gen_batch.next_state))  # [trajectory length * parallel size, state size]
        gen_batch_old_log_prob = trans_shape_func(
            torch.stack(gen_batch.log_prob))  # [trajectory length * parallel size, 1]
        gen_batch_mask = trans_shape_func(torch.stack(gen_batch.mask))  # [trajectory length * parallel size, 1]
        gen_batch_done = trans_shape_func(torch.stack(gen_batch.done))
        # grad_collect_func = lambda d: torch.cat([grad.view(-1) for grad in torch.autograd.grad(d, self.D.parameters(), retain_graph=True)]).unsqueeze(0)
        ####################################################
        # update discriminator
        ####################################################
        for expert_batch_state, expert_batch_action in self.expert_data_loader:
            gen_r = self.D(gen_batch_state, gen_batch_action)
            expert_r = self.D(expert_batch_state.to(device), expert_batch_action.to(device))

            # label smoothing for discriminator
            expert_labels = torch.ones_like(expert_r)
            gen_labels = torch.zeros_like(gen_r)

            if self.config.discriminator_use_label_smoothing:
                smoothing_rate = self.config.discriminator_label_smooth_rate
                expert_labels *= (1 - smoothing_rate)
                gen_labels += torch.ones_like(gen_r) * smoothing_rate

            e_loss = self.discriminator_func(expert_r, expert_labels)
            g_loss = self.discriminator_func(gen_r, gen_labels)
            d_loss = e_loss + g_loss

            # """ WGAN with Gradient Penalty"""
            # d_loss = gen_r.mean() - expert_r.mean()
            # differences_batch_state = gen_batch_state[:expert_batch_state.size(0)] - expert_batch_state
            # differences_batch_action = gen_batch_action[:expert_batch_action.size(0)] - expert_batch_action
            # alpha = torch.rand(expert_batch_state.size(0), 1)
            # interpolates_batch_state = gen_batch_state[:expert_batch_state.size(0)] + (alpha * differences_batch_state)
            # interpolates_batch_action = gen_batch_action[:expert_batch_action.size(0)] + (alpha * differences_batch_action)
            # gradients = torch.cat([x for x in map(grad_collect_func, self.D(interpolates_batch_state, interpolates_batch_action))])
            # slopes = torch.norm(gradients, p=2, dim=-1)
            # gradient_penalty = torch.mean((slopes - 1.) ** 2)
            # d_loss += 10 * gradient_penalty

            self.optimizer_discriminator.zero_grad()
            d_loss.backward()
            self.optimizer_discriminator.step()

            self.scheduler_discriminator.step()

        self.writer.add_scalar('train/loss/d_loss', d_loss.item(), epoch)
        self.writer.add_scalar("train/loss/e_loss", e_loss.item(), epoch)
        self.writer.add_scalar("train/loss/g_loss", g_loss.item(), epoch)
        self.writer.add_scalar('train/reward/expert_r', expert_r.mean().item(), epoch)
        self.writer.add_scalar('train/reward/gen_r', gen_r.mean().item(), epoch)

        with torch.no_grad():
            gen_batch_value = self.V(gen_batch_state)
            gen_batch_next_value = self.V(gen_batch_next_state)
            gen_batch_reward = self.D(gen_batch_state, gen_batch_action)

        gen_batch_advantage, gen_batch_return = estimate_advantages(gen_batch_reward, gen_batch_mask,
                                                                    gen_batch_value, gen_batch_next_value, self.config.gae_gamma,
                                                                    self.config.gae_tau,
                                                                    gen_batch_done)

        ####################################################
        # update policy by ppo [mini_batch]
        ####################################################
        ppo_optim_epochs = self.config.ppo_ppo_optim_epochs
        ppo_mini_batch_size = self.config.ppo_ppo_mini_batch_size
        gen_batch_size = gen_batch_state.shape[0]
        optim_iter_num = int(math.ceil(gen_batch_size / ppo_mini_batch_size))

        for _ in range(ppo_optim_epochs):
            perm = torch.randperm(gen_batch_size)

            for i in range(optim_iter_num):
                ind = perm[slice(i * ppo_mini_batch_size,
                                 min((i + 1) * ppo_mini_batch_size, gen_batch_size))]
                mini_batch_state, mini_batch_action, mini_batch_next_state, mini_batch_advantage, mini_batch_return, \
                mini_batch_old_log_prob = gen_batch_state[ind], gen_batch_action[ind], gen_batch_next_state[ind], \
                                          gen_batch_advantage[ind], gen_batch_return[ind], gen_batch_old_log_prob[ind]

                v_loss, p_loss = ppo_step(self.P, self.V, self.optimizer_policy, self.optimizer_value,
                                          states=mini_batch_state,
                                          actions=mini_batch_action,
                                          next_states=mini_batch_next_state,
                                          returns=mini_batch_return,
                                          old_log_probs=mini_batch_old_log_prob,
                                          advantages=mini_batch_advantage,
                                          ppo_clip_ratio=self.config.ppo_clip_ratio,
                                          value_l2_reg=self.config.value_l2_reg)
                
                self.writer.add_scalar('train/loss/p_loss', p_loss, epoch)
                self.writer.add_scalar('train/loss/v_loss', v_loss, epoch)

        print(f" Training episode:{epoch} ".center(80, "#"))
        print('gen_r:', gen_r.mean().item())
        print('expert_r:', expert_r.mean().item())
        print('d_loss', d_loss.item())

    # def eval(self, epoch):
    #     self.P.eval()
    #     self.D.eval()
    #     self.V.eval()

    #     gen_batch = self.P.collect_samples(self.config.ppo_sample_batch_size)
    #     gen_batch_state = torch.stack(gen_batch.state)
    #     gen_batch_action = torch.stack(gen_batch.action)

    #     gen_r = self.D(gen_batch_state, gen_batch_action)
    #     for expert_batch_state, expert_batch_action in self.expert_data_loader:
    #         expert_r = self.D(expert_batch_state.to(device), expert_batch_action.to(device))

    #         print(f" Evaluating episode:{epoch} ".center(80, "-"))
    #         print('validate_gen_r:', gen_r.mean().item())
    #         print('validate_expert_r:', expert_r.mean().item())

    #     self.writer.add_scalar("validate/reward/gen_r", gen_r.mean().item(), epoch)
    #     self.writer.add_scalar("validate/reward/expert_r", expert_r.mean().item(), epoch)

    def save_model(self, save_path,i):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # dump model from pkl file
        # torch.save((self.D, self.P, self.V), f"{save_path}/{self.exp_name}.pt")
        torch.save(self.D, f"{save_path}/{str(i)}_Discriminator.pt")
        torch.save(self.P, f"{save_path}/{str(i)}_JointPolicy.pt")
        torch.save(self.V, f"{save_path}/{str(i)}_Value.pt")

    def load_model(self, model_path):
        # load entire model
        # self.D, self.P, self.V = torch.load((self.D, self.P, self.V), f"{save_path}/{self.exp_name}.pt")
        self.D = torch.load(f"{model_path}/Discriminator.pt", map_location=device)
        self.P = torch.load(f"{model_path}/JointPolicy.pt", map_location=device)
        self.V = torch.load(f"{model_path}/Value.pt", map_location=device)
