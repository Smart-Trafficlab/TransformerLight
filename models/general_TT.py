import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .DT.network_agent import NetworkAgent
import copy
from .TT.gpt import GPT
import pickle
import torch.nn.functional as F


NUM_DEVICE=3
class GeneralTrajectoryTransformer(NetworkAgent):

    def load_data(self):
        path1 = self.dic_path["PATH_TO_MEMORY"]
        print(path1)

        with open(path1, "rb") as f:
            memory = pickle.load(f)

        return memory

    def build_network(self):
        observation_dim = 132
        action_dim = 4
        block_size = observation_dim + action_dim + 1 - 1
        seq = 1
        transition_dim = (observation_dim + action_dim + 2) * seq
        _state, _action, _, _reward = self.prepare_samples(self.load_data())
        # ==== shuffle the samples ============
        random_index = np.random.permutation(len(_action))
        _state[0] = _state[0][random_index, :, :]
        _action = np.array(_action)[random_index]
        _reward = np.array(_reward)[random_index]
        _reward = np.abs(_reward.min()) + _reward
        _state[0] = np.abs(_state[0].min()) + _state[0]
        vocab_size = int(np.ceil(max(_state[0].max(), _reward.max())))
        network = GPT(
            n_head=4, attn_pdrop=0.1, resid_pdrop=0.1,
            block_size=block_size, n_embd=256 * 4, embd_pdrop=0.1, n_layer=4,  action_weight=5, reward_weight=1, value_weight=1,
            vocab_size=vocab_size,
            transition_dim=transition_dim,
            observation_dim=observation_dim,
            action_dim=action_dim
        )
        
        return network

    def choose_action(self, states):
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "new_phase":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        used_feature.remove("new_phase")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        cur_states_len = len(states)
        batch_Xs1 = torch.tensor(state_input, dtype=torch.long).squeeze(0).reshape(-1, 12*self.num_feat)
        actions = torch.zeros((cur_states_len, 4), dtype=torch.long)
        tokens = torch.cat([batch_Xs1, actions], axis=1)
        seq, _ = self.model(tokens)
        q_values = np.zeros([seq.shape[0], 4])
        # seq[num_sec * trajectory * probs]
        # trajectory[-5:-1] is action
        # vocab index is the value of action 
        for i in range(seq.shape[0]):
            for j in range(4):
                q_values[i, (-j+1)] = seq[i, -(j+2), :].argmax()
     
        action = np.argmax(q_values, axis=1)
        self.time_step += 1
        return action

    def prepare_samples(self, memory):
        state, action, next_state, p_reward, ql_reward = memory
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        memory_size = len(action)
        _state = [[], None]
        _next_state = [[], None]
        for feat_name in used_feature:
            if feat_name == "new_phase":
                _state[1] = np.array(state[feat_name])
                _next_state[1] = np.array(next_state[feat_name])
            else:
                _state[0].append(np.array(state[feat_name]).reshape(memory_size, 12, -1))
                _next_state[0].append(np.array(next_state[feat_name]).reshape(memory_size, 12, -1))
                
        # ========= generate reaward information ===============
        if "pressure" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys():
            my_reward = p_reward
        else:
            my_reward = ql_reward
        
        return [np.concatenate(_state[0], axis=-1), _state[1]], action, [np.concatenate(_next_state[0], axis=-1), _next_state[1]], my_reward

    def train_network(self, memory):
        self.observation_dim = 132
        self.action_dim = 4
        _state, _action, _, _reward = self.prepare_samples(memory)
        # ==== shuffle the samples ============
        random_index = np.random.permutation(len(_action))
        _state[0] = _state[0][random_index, :, :]
        _action = np.array(_action)[random_index]
        _reward = np.array(_reward)[random_index]
        _reward = np.abs(_reward.min()) + _reward
        _state[0] = np.abs(_state[0].min()) + _state[0]
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(_action))
        batch_size =256
        num_batch = int(np.floor((len(_action) / batch_size)))
  
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        for epoch in range(epochs):
            for ba in range(1):
                batch_Xs1 = torch.tensor([_state[0][ba*batch_size:(ba+1)*batch_size, :, :]], dtype=torch.long).\
                                squeeze(0).reshape(batch_size, 12*self.num_feat)
                
                batch_r = torch.tensor(_reward[ba*batch_size:(ba+1)*batch_size], dtype=torch.long).view(-1,1)
                batch_a = F.one_hot(torch.tensor(_action[ba*batch_size:(ba+1)*batch_size]), 4).type(torch.long)

                tokens = torch.cat([batch_Xs1, batch_a, batch_r], axis=1)
                cur_tokens = torch.tensor(tokens.clone()[:, 0:136], dtype=torch.long)
                target_tokens = torch.tensor(tokens.clone()[:, 1:137], dtype=torch.long)
                pre_seq, loss = self.model(cur_tokens, target_tokens)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
                print("===== Epoch {} | Batch {} / {} | Loss {}".format(epoch, ba, num_batch, loss))


    