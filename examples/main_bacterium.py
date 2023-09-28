from os import path
from os.path import dirname, join, abspath
import numpy as np
from scipy.io import savemat
import signal
from mpi4py import MPI
from pyrep.common.arguments_v0 import get_args
import os
from pyrep.baselines.bacterium.drone_bacterium_env_camera import Drone_Env_bacterium

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# get the params
args = get_args()

if args.field_size == 10:
    env_name = join(dirname(abspath(__file__)), 'RL_drone_field_10x10.ttt')
if args.field_size == 15:
    env_name = join(dirname(abspath(__file__)), 'RL_drone_field_15x15.ttt')

num_agents = args.n_agents
# create multiagent environment
env = Drone_Env_bacterium(args, rank, env_name,num_agents)

save_path = './'+ 'flock_analysis/camera/'+args.scenario_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

    # signal.pause()
while 1: 
    if env.done:
        print("start saving data")
        pos_x = comm.gather(env.pos_x, root=0)
        pos_y = comm.gather(env.pos_y, root=0)
        pos_z = comm.gather(env.pos_z, root=0)
        c_sum = comm.gather(env.c_sum, root=0)

        if rank ==0:
            savemat(save_path+'/storePX.mat', mdict={'arr': pos_x})
            savemat(save_path+'/storePY.mat', mdict={'arr': pos_y})
            savemat(save_path+'/storePZ.mat', mdict={'arr': pos_z})

            savemat(save_path+'/storeAttachPoints.mat', mdict={'arr': env.attach_points})
            savemat(save_path+'/storeConcentration_sum.mat', mdict={'arr': c_sum})

            print("fail number", env.fail_num)
            print("finish data saving")

        env.shutdown()
        break

    env.step()  








