# from pyrep.policies.maddpg_drone_att_mpi_V0 import MADDPG
# from pyrep.policies.maddpg_drone_att_mpi_orca import MADDPG
# from pyrep.policies.maddpg_drone_att_attention import MADDPG
from pyrep.policies.maddpg_drone_att_demowp import MADDPG
# from pyrep.common.arguments_v0 import get_args
from pyrep.common.arguments_mappo import get_args
import numpy as np
from pyrep.baselines.mappo.mappo_mpe import MAPPO_MPE


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num, ' Trainable: ', trainable_num)


if __name__ == '__main__':
    # get the params
    # args = get_args()
    # args.n_agents = 4

    # args.lr_actor=1e-4 
    # args.lr_critic=1e-3 

    # args.high_action = 1
    # args.load_buffer = False
    # args.obs_shape = [2+2+4*(args.n_agents-1)+2*args.n_agents for _ in range(args.n_agents)]  # observation space

    # args.scenario_name = "cal_params"

    # args.save_dir = "./" + args.scenario_name + "/model_drone{}".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)+'/'+ 'env'
    
    # args.use_gpu = False
    # # print(args.obs_shape)
    # # assert(args.obs_shape[0]==82)

    # action_shape = []        
    # for _ in range(args.n_agents):
    #     action_shape.append(2)
    # args.action_shape = action_shape[:args.n_agents]  # action space
    # # print(args.action_shape)
    # assert(args.action_shape[0]==2) 

    # agents = [MADDPG(args,i) for i in range(args.n_agents)]


    # MAPPO
    args = get_args()
    args.n_agents = 6

    args.high_action = 1
    args.load_buffer = False
    args.obs_shape = [2+2+4*(args.n_agents-1)+2*args.n_agents for _ in range(args.n_agents)]  # observation space

    args.scenario_name = "cal_params"
    args.save_dir = "./" + args.scenario_name + "/model_drone{}".format(args.n_agents)+'/'+'field_size{}'.format(args.field_size)+'/'+ 'env'
    
    args.use_gpu = False
    # print(args.obs_shape)
    # assert(args.obs_shape[0]==82)
    action_shape = []        
    for _ in range(args.n_agents):
        action_shape.append(5)
    args.action_shape = action_shape[:args.n_agents]  # action space
    # print(args.action_shape)
    assert(args.action_shape[0]==5) 

    # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
    args.obs_dim = args.obs_shape[0]  # The dimensions of an agent's observation space
    args.action_dim = args.action_shape[0]  # The dimensions of an agent's action space
    args.state_dim = np.sum(args.obs_shape)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
    args.N = args.n_agents

    # Create N agents
    agent_n = MAPPO_MPE(args)
    
    get_parameter_number(agent_n)