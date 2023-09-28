from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.io import savemat

from pyrep.common.optimizers import CEMOptimizer
from pyrep.policies.PETS import PETS_model
from tqdm import trange

import torch
from torch.nn import functional as F

TORCH_DEVICE = torch.device('cuda')

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


class Controller:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        pass

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, obs, t, log_pred_data=False):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, params, env):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(params)
        
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.max_x = env.field_size
        self.max_y = env.field_size
        self.num_a = env.num_a
        self.goals = env.goals
        # self.update_fns = params.update_fns
        self.per = params.per
        
        self.prop_mode = params.prop_mode
        self.npart = params.npart #num of particles for cem
        self.ign_var = params.ign_var  

        self.opt_mode = params.opt_mode
        self.plan_hor = params.plan_hor
        self.num_nets = params.num_nets  #emsemble models
        self.epsilon = params.epsilon,
        self.alpha = params.alpha
        self.epochs = params.epochs
        self.max_iters = params.max_iters
        self.popsize = params.popsize
        self.num_elites = params.num_elites

        self.model_3d_in = params.model_3d_in
        self.model_3d_out = params.model_3d_out
    
        self.save_all_models = False
        self.log_traj_preds = False
        self.log_particles = False

        # Perform argument checks
        assert self.opt_mode == 'CEM'
        assert self.prop_mode == 'TSinf' #'only TSinf propagation mode is supported'
        assert self.npart % self.num_nets == 0, "Number of particles must be a multiple of the ensemble size."

        # Create action sequence optimizer
        self.optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            cost_function=self._compile_cost,
            epsilon = self.epsilon,
            alpha = self.alpha,
            max_iters = self.max_iters ,
            popsize = self.popsize,
            num_elites = self.num_elites
        )

        # Controller state variables
        self.has_been_trained =  False
        self.ac_buf = np.array([]).reshape(0, self.dU)
        #sol: [act_dim*plan_hor,]
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

        # Set up pytorch model
        self.load_model = params.load_model
        self.train_in = np.array([]).reshape(0, self.dU + self.dO)
        self.train_targs = np.array([]).reshape(0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1])
        
        self.model = self.nn_constructor(self.num_nets,self.model_3d_in,self.model_3d_out)
        self.epoch_sum = 0


    def train(self, obs_trajs, acs_trajs, next_obs_trajs, logger):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.

        Returns: None.
        """

        if not self.has_been_trained:
            #in this part, actions are not the true action, is the normalized reference goal position
            new_train_in, new_train_targs = [], []
            
            #action is real relative distance not normalized one
            new_train_in.append(np.concatenate([obs_trajs, acs_trajs], axis=-1))
            new_train_targs.append(self.targ_proc(obs_trajs, next_obs_trajs))

            self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
            self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

            ###########################
            # Train the pytorch model
            self.model.net.fit_input_stats(self.train_in)

            idxs = np.random.randint(self.train_in.shape[0], size=[self.model.net.num_nets, self.train_in.shape[0]])

            epochs = self.epochs

            # TODO: double-check the batch_size for all env is the same
            batch_size = 100

            epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
            num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

            for i in epoch_range:
                train_loss = 0 
                validate_loss = 0

                for batch_num in range(num_batch):
                    batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

                    loss = 0.01 * (self.model.net.max_logvar.sum() - self.model.net.min_logvar.sum())
                    loss += self.model.net.compute_decays()

                    # TODO: move all training data to GPU before hand
                    train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
                    train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()

                    mean, logvar = self.model.net(train_in, ret_logvar=True)
                    inv_var = torch.exp(-logvar)

                    train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                    # train_losses = (mean - train_targ) ** 2
                    train_losses = train_losses.mean(-1).mean(-1).sum()
                    # Only taking mean over the last 2 dimensions
                    # The first dimension corresponds to each model in the ensemble

                    loss += train_losses
                    train_loss += train_losses.item()

                    self.model.optim.zero_grad()
                    loss.backward()
                    self.model.optim.step()

                logger.add_scalar('Train_iter_offline/Training loss', train_loss/num_batch, i)
                print('Offline: step:', i, '\ttraining acc:', train_loss/num_batch)

                idxs = shuffle_rows(idxs)

                val_in = torch.from_numpy(self.train_in[idxs[:500]]).to(TORCH_DEVICE).float()
                val_targ = torch.from_numpy(self.train_targs[idxs[:500]]).to(TORCH_DEVICE).float()

                mean, _ = self.model.net(val_in)
                mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
                validate_loss += mse_losses.item()
                
                logger.add_scalar('Validation_iter_offline/Validation loss', validate_loss, i)
                print('Offline: step:', i, '\ttest acc:', validate_loss)

                # if i%10:
                #     self.model.save_model(i)
            ###########################

            self.has_been_trained = True
            return 
        else:
            # Construct new training points and add to training set
            #true action, normalized observations
            new_train_in, new_train_targs = [], []
            for obs, acs, obs_next in zip(obs_trajs, acs_trajs, next_obs_trajs):
                new_train_in.append(np.concatenate([obs, acs], axis=-1))
                new_train_targs.append(self.targ_proc(obs, obs_next))
            self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
            self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        self.has_been_trained = True

        # Train the pytorch model
        self.model.net.fit_input_stats(self.train_in)

        idxs = np.random.randint(self.train_in.shape[0], size=[self.model.net.num_nets, self.train_in.shape[0]])

        epochs = self.epochs

        # TODO: double-check the batch_size for all env is the same
        batch_size = 100

        epoch_range = epochs
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for i in range(self.epoch_sum, self.epoch_sum+epoch_range):
            train_loss = 0 
            validate_loss = 0

            for batch_num in range(num_batch):
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

                loss = 0.01 * (self.model.net.max_logvar.sum() - self.model.net.min_logvar.sum())
                loss += self.model.net.compute_decays()

                # TODO: move all training data to GPU before hand
                train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
                train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()

                mean, logvar = self.model.net(train_in, ret_logvar=True)
                inv_var = torch.exp(-logvar)

                train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                # train_losses = (mean - train_targ) ** 2
                train_losses = train_losses.mean(-1).mean(-1).sum()
                # Only taking mean over the last 2 dimensions
                # The first dimension corresponds to each model in the ensemble

                loss += train_losses
                train_loss += train_losses.item()

                self.model.optim.zero_grad()
                loss.backward()
                self.model.optim.step()
            
            logger.add_scalar('Train_iter_online/Training loss', train_loss/num_batch, i)

            idxs = shuffle_rows(idxs)

            val_in = torch.from_numpy(self.train_in[idxs[:500]]).to(TORCH_DEVICE).float()
            val_targ = torch.from_numpy(self.train_targs[idxs[:500]]).to(TORCH_DEVICE).float()

            mean, _ = self.model.net(val_in)
            mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
            validate_loss += mse_losses.item()
           
            logger.add_scalar('Validation_iter_online/Validation loss', validate_loss, i)

            if i%10==0:
                self.model.save_model(i)

        self.epoch_sum+=epoch_range


    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()

        # for update_fn in self.update_fns:
        #     update_fn()


    def act(self, obs, t, goal, log_pred_data=False):
        """Returns the action that this controller would take at time t given observation obs.
           for trajectory tracking, we have to iter the goals

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        # if not self.has_been_trained:
        #     return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            # print("action_buffer",self.ac_buf)
            # print("action:", action)
            return action,self.soln_elites,self.store_top_s,self.store_bad_s

        self.sy_cur_obs = obs

        # print("current pos", self.sy_cur_obs[0]*self.max_x, self.sy_cur_obs[1], self.sy_cur_obs[2]*self.max_z)
        # print("goal:", goal.shape)
        #[soldim,] [10,soldim]
        soln, self.soln_elites,self.store_top_s,self.store_bad_s = self.optimizer.obtain_solution(self.prev_sol, self.init_var, goal)
        # print("solutions",soln)
        # print(self.store_top_s.shape)
        
        assert(self.store_top_s.shape[0]==self.plan_hor+1 and self.store_top_s.shape[1]==5 and self.store_top_s.shape[2]==self.num_a*2)
        #zeros part may be replaced by the (self.act_high+self.act_low)/2
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        #only store one solution, thus will update each time
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)
        # print("action_buffer",self.ac_buf)

        return self.act(obs, t, goal)

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        # TODO: implement saving model for pytorch
        # self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []

    def dis_array(self, arr1, arr2):
        assert(arr1.shape == arr2.shape)
        return np.sqrt(np.sum((arr1-arr2)**2,axis=1,keepdims=1))


    def obs_mpc_cost_fn(self, obs_seq):
        obs_seq1 = obs_seq.clone().detach().cpu().numpy()
        obs_sep_expand = []
        for i in range(obs_seq1.shape[-1]//4):
            obs_sep_expand.append(obs_seq1[:,i*2:i*2+2])

        dis_expand = []
        for j in range(len(obs_sep_expand)-1):
            for m in range(len(obs_sep_expand)-j-1):
                dis_expand.append(self.dis_array(obs_sep_expand[j],obs_sep_expand[m+j+1]))
        
        dis_all = np.concatenate(dis_expand,axis=1)
        collison_arr = dis_all < 0.5

        collison_arr = np.any(collison_arr,axis=1)
        assert(collison_arr.shape[0]==obs_seq.shape[0])

        return torch.from_numpy(collison_arr*10).to(TORCH_DEVICE)

    # template mpc cost function. used for sampling action sequences
    def default_mpc_cost_fn(self, obs_seq, goal_seq):
        # (N, obsdim) (n,3)
        # this std is some scaled version of the observation standard deviation
        # std = model_out_seq.next_obs_sigma[:, :1, 0]  # (N, 1, obsdim) TODO non deterministic
        obs_seq1 = obs_seq[:,:obs_seq.shape[-1]//2].clone().detach()
        goal_seq1 = goal_seq

        ######################################
        normalized = torch.abs(obs_seq1*self.max_x - torch.flatten(goal_seq1))

        return normalized.sum(1)  # (N,)
    

    @torch.no_grad()
    def _compile_cost(self, ac_seqs, goal):

        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # Before, ac seqs has dimension (400, 25) which are pop size and sol dim coming from CEM
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        #  After, ac seqs has dimension (400, 25, 2)

        transposed = ac_seqs.transpose(0, 1)
        # Then, (25, 400, 2)

        expanded = transposed[:, :, None]
        # Then, (25, 400, 1, 2)

        tiled = expanded.expand(-1, -1, self.npart, -1)
        # Then, (25, 400, 20, 2)

        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)
        # Then, (25, 8000, 2)
        goal = torch.from_numpy(goal).float().to(TORCH_DEVICE)
        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        costs = torch.zeros(nopt, self.npart, device=TORCH_DEVICE)
        
        store_states = [cur_obs[:,:self.num_a*2].view(-1,self.npart,self.num_a*2).mean(dim=1).clone().detach().cpu().numpy()]

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            #[8000, dim1] 
            # cost = self.default_mpc_cost_fn(next_obs[:,:3], goal[0,t,:]) + self.ac_cost_fn(cur_acs)

            #[nopt * npart, 2]->[nopt * npart]
            cost = self.default_mpc_cost_fn(next_obs, goal) + self.obs_mpc_cost_fn(next_obs) + self.ac_cost_fn(cur_acs)

            s_t = next_obs[:,:self.num_a*2].view(-1,self.npart,self.num_a*2).mean(dim=1).clone().detach().cpu().numpy() #[400,20,3]
            assert(s_t.shape[0]==nopt and s_t.shape[1]==self.num_a*2)

            store_states.append(s_t) #[400,2]
            ####
            #[nopt,npart]
            cost = cost.view(-1, self.npart)

            costs += cost
            cur_obs = next_obs

        # Replace nan with high cost
        costs[costs != costs] = 1e6

        #return [nopt,]

        return costs.mean(dim=1).detach().cpu().numpy(), store_states

    def _validate_prediction(self,obs,acs,next_obs):
        obs = torch.from_numpy(obs).float().to(TORCH_DEVICE)
        obs = obs[None]
        acs = torch.from_numpy(acs).float().to(TORCH_DEVICE)
        acs = acs[None]
        next_obs = torch.from_numpy(next_obs).float().to(TORCH_DEVICE)

        proc_obs = self.obs_preproc_3d(obs)
        
        inputs = torch.cat((proc_obs, acs), dim=-1)

        mean, var = self.model.net(inputs)
        predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()
       
        
        prediction_s = self.obs_postproc(obs, predictions)
        loss = F.mse_loss(prediction_s, next_obs)

        return loss.item() 

    def _predict_next_obs(self, obs, acs):
        proc_obs = obs

        assert self.prop_mode == 'TSinf'

        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        mean, var = self.model.net(inputs)
        predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return self.obs_postproc(obs, predictions)

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [8000, 5] in case of proc_obs
        #[400,1,20,5]
        reshaped = mat.view(-1, self.model.net.num_nets, self.npart // self.model.net.num_nets, dim)
        
        transposed = reshaped.transpose(0, 1)
        # After, [1, 400, 20, 5]

        reshaped = transposed.contiguous().view(self.model.net.num_nets, -1, dim)
       
        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.model.net.num_nets, -1, self.npart // self.model.net.num_nets, dim)
        
        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped

#################

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def ac_cost_fn(acs):
        #this cost is to constraint reference goal is not too far from the load
        return 0.25 * (acs**2).sum(dim=1)

    # definitions of different neural network models used in MPC
    def nn_constructor(self, num_nets, model_in, model_out):

        #initialize nerworks here
        ensemble_size = num_nets

        model = PETS_model(ensemble_size, model_in, model_out,load_model=self.load_model)
        # * 2 because we output both the mean and the variance

        # model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


        


        



