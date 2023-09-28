from torch import nn
from pyrep.networks.neural_nets import PtModel
import os
from time import localtime, strftime
import torch

TORCH_DEVICE = torch.device('cuda')

class PETS_model(nn.Module):
    def __init__(self, ensemble_size, model_in, model_out, load_model=False):
        super(PETS_model,self).__init__()
        self.net = PtModel(ensemble_size, model_in, model_out * 2).to(TORCH_DEVICE)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.model_path = os.path.join("/home/wawa/RL_transport_3D/PETS_model_3d",strftime("%Y-%m-%d--%H:%M:%S", localtime()))
        os.makedirs(self.model_path, exist_ok=True)

        if load_model:
            load_path = "/home/wawa/RL_transport_3D/PETS_model_3d"
            if os.path.exists(load_path + '/params.pkl'):
                self.initialise_networks(load_path+'/params.pkl')
                print('Agent successfully loaded PETS_network: {}'.format(load_path + '/params.pkl'))
            else:
                print('Failed to load model from PETS')

    def initialise_networks(self, path):
        
        checkpoint = torch.load(path) # load the torch data

        self.net.load_state_dict(checkpoint['PETS_params'])    # actor parameters
        self.optim.load_state_dict(checkpoint['PETS_optim_params']) # critic optimiser state
        
    def save_model(self, train_step):
        num = str(train_step)
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        save_dict = {'PETS_params' : self.net.state_dict(),
                    'PETS_optim_params' : self.optim.state_dict()}

        torch.save(save_dict, self.model_path + '/' + num + '_params.pkl') 
