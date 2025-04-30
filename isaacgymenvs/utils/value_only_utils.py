import gym
import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.experience import ExperienceBuffer
from torch import nn
from isaacgymenvs.learning.architectures.lipschitz_mlp.lipmlp import LipschitzMLP

class TransformerValueNet(nn.Module):
    def __init__(self, net=None, input_size=None, normalize_input=True, normalize_value=True):
        super().__init__()
        assert net != None
        self.net = net
        self.input_size = input_size
        self.value_size = 1
        self.final_layer = nn.Linear(net.nlatent, self.value_size)
        self.normalize_input = normalize_input
        self.normalize_value = normalize_value
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd((self.input_size,))
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,))
    
    def norm_obs(self, observation):
        with torch.no_grad():
            obs_shape = observation.shape
            observation = observation.view(-1, obs_shape[-1])
            norm_observation = self.running_mean_std(observation) if self.normalize_input else observation
            norm_observation = norm_observation.view(*obs_shape)
            return norm_observation

    def unnorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, unnorm=True) if self.normalize_value else value

    def forward(self, input_dict, activation=None):
        if self.normalize_input:
            features = self.norm_obs(input_dict['features'])
        frame_indices = deltas_to_indices(input_dict['frame_deltas'])
        net_input_dict = dict(
            features=features,
            frame_indices=frame_indices
        )
        out = self.net(net_input_dict)
        out = self.final_layer(out)
        if activation != None:
            raise NotImplementedError("value network must not have activation layer")
        loss_dict = dict()
        return out, loss_dict

    def set_dropout_train(self,):
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.training = True

    def set_dropout_eval(self,):
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.training = False


class TransformerMLPValueNet(nn.Module):
    def __init__(self, trans_net=None, mlp_net=None, input_size=None, normalize_input=True, normalize_value=True):
        super().__init__()
        assert trans_net != None
        assert mlp_net != None
        self.net = trans_net
        self.input_size = input_size
        self.value_size = 1
        self.final_layer = mlp_net
        self.normalize_input = normalize_input
        self.normalize_value = normalize_value
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd((self.input_size,))
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,))
    

    def norm_obs(self, observation):
        with torch.no_grad():
            obs_shape = observation.shape
            observation = observation.view(-1, obs_shape[-1])
            norm_observation = self.running_mean_std(observation) if self.normalize_input else observation
            norm_observation = norm_observation.view(*obs_shape)
            return norm_observation
    
    def unnorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, unnorm=True) if self.normalize_value else value

    def forward(self, input_dict, activation=None):
        if self.normalize_input:
            input_dict['features'] = self.norm_obs(input_dict['features'])
        input_dict['frame_indices'] = deltas_to_indices(input_dict['frame_deltas'])
        out = self.net(input_dict)
        input_dict['features'] = out
        out, loss_dict = self.final_layer(input_dict)
        if activation != None:
            raise NotImplementedError("value network must not have activation layer")
        return out, loss_dict

    def set_dropout_train(self,):
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.training = True

    def set_dropout_eval(self,):
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.training = False

class MLPValueNet(nn.Module):
    def __init__(self, net=None, input_size=None, normalize_input=True, normalize_value=True):
        super().__init__()
        assert net != None
        self.net = net
        self.input_size = input_size
        self.value_size = 1
        self.normalize_input = normalize_input
        self.normalize_value = normalize_value
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd((self.input_size,))
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,))
    

    def norm_obs(self, observation):
        with torch.no_grad():
            obs_shape = observation.shape
            observation = observation.view(-1, obs_shape[-1])
            norm_observation = self.running_mean_std(observation) if self.normalize_input else observation
            norm_observation = norm_observation.view(*obs_shape)
            return norm_observation

    def unnorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, unnorm=True) if self.normalize_value else value

    def forward(self, input_dict, activation=None):
        if self.normalize_input:
            input_dict['features'] = self.norm_obs(input_dict['features'])
        input_dict['frame_indices'] = deltas_to_indices(input_dict['frame_deltas'])
        out, loss_dict = self.net(input_dict)
        if activation != None:
            raise NotImplementedError("value network must not have activation layer")
        return out, loss_dict

class LongTermValueNetKinematic(nn.Module):
    def __init__(self, net=None, normalize_input=True, normalize_value=True):
        super().__init__()
        assert net != None
        self.net = net
        self.value_size = 1
        self.final_layer = nn.Linear(net.latent_dim, self.value_size)
        self.normalize_input = normalize_input
        self.normalize_value = normalize_value
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,))

    def norm_obs(self, observation):
        B, T, _ = observation.shape
        observation = observation.view(B, T, -1, 9)
        observation = observation.view(B * T, -1, 9)
        norm_observation = self.net.norm_features(observation, freeze=True)
        norm_observation = norm_observation.view(B, T, -1)
        return norm_observation 

    def unnorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, unnorm=True) if self.normalize_value else value

    def forward(self, input_dict, activation=None):
        if self.normalize_input:
            input_dict['features'] = self.norm_obs(input_dict['features'])
        input_dict['frame_indices'] = deltas_to_indices(input_dict['frame_deltas'])
        out = self.net.encode(features=input_dict['features'], indices=input_dict['frame_indices'], out_latent_only=True)
        out = self.final_layer(out)
        if activation != None:
            raise NotImplementedError("value network must not have activation layer")
        loss_dict = dict()
        return out, loss_dict

    def set_dropout_train(self,):
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.training = True

    def set_dropout_eval(self,):
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.training = False

class EpisodeExperienceBuffer(ExperienceBuffer):
    def __init__(self, env_info, algo_info, device):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get('agents', 1)
        self.num_actors = algo_info['num_actors']
        self.obs_base_shape = (self.num_agents * self.num_actors, )
        self.tensor_dict = {}
        self.tensor_buffer_dict = {} # for temporary saves
        self._init_from_env_info(self.env_info)
        self.curr_idx = 0

    def _init_from_env_info(self, env_info):
        obs_base_shape = self.obs_base_shape
        self.tensor_dict['obses'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
        self.tensor_buffer_dict['obses'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
        val_space = gym.spaces.Box(low=0, high=1,shape=(env_info.get('value_size',1),))
        self.tensor_dict['returns'] = self._create_tensor_from_space(val_space, obs_base_shape)

    def update_data(self, name, batch_ids, val):
        raise NotImplementedError("Depreicated usage")
    
    def reset_data(self):
        # IMPORTANT : don't erase buffer data
        self.curr_idx = 0
    
    def push_data_list(self, n, val_dict):
        s_idx = self.curr_idx
        e_idx = s_idx + n
        if e_idx > self.obs_base_shape[0]:
            v_idx = n - (e_idx - self.obs_base_shape[0])
            e_idx = self.obs_base_shape[0]
            for name, val in val_dict.items():
                if type(val) is dict:
                    for k,v in val.items():
                        self.tensor_dict[name][k][s_idx:e_idx] = v[:v_idx]
                else:
                    self.tensor_dict[name][s_idx:e_idx] = val[:v_idx]
        else:
            for name, val in val_dict.items():
                if type(val) is dict:
                    for k,v in val.items():
                        self.tensor_dict[name][k][s_idx:e_idx] = v
                else:
                    self.tensor_dict[name][s_idx:e_idx] = val
        self.curr_idx = e_idx

    def update_buffer(self, name, batch_ids, val):
        if type(val) is dict:
            for k,v in val.items():
                self.tensor_buffer_dict[name][k][batch_ids] = v
        else:
            self.tensor_buffer_dict[name][batch_ids] = val

    def get_list(self, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = vd
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = v
        
        return res_dict

    def get_buffer(self, name, batch_ids):
        v = self.tensor_buffer_dict[name]
        if type(v) is dict:
            ret = dict()
            for kd, vd in v.items():
                ret[kd] = vd[batch_ids]
        else:
            ret = v[batch_ids]
        return ret
    
    def is_full(self):
        return self.curr_idx == self.obs_base_shape[0]



@torch.jit.script
def deltas_to_indices(deltas):
    # type: (Tensor) -> Tensor
    deltas = torch.cat([torch.zeros_like(deltas[:, :1]), deltas], dim=-1) 
    indices = torch.cumsum(deltas, dim=-1)
    return indices
