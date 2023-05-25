import numpy as np
from tensorflow.keras.layers import Layer, Reshape
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import random
import os
from .agent import Agent
import traceback


class ActorCriticNetworkAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0"):
        super(ActorCriticNetworkAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id=intersection_id)

        # ===== check num actions == num phases ============
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.num_phases = len(dic_traffic_env_conf["PHASE"])
        # self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))

        self.memory = self.build_memory()
        self.cnt_round = cnt_round

        self.num_lane = dic_traffic_env_conf["NUM_LANE"]
        # self.max_lane = dic_traffic_env_conf["MAX_LANE"]
        self.phase_map = dic_traffic_env_conf["PHASE_MAP"]

        self.len_feat = self.cal_input_len()
        self.num_feat = int(self.len_feat/12)
        self.min_q_weight = dic_traffic_env_conf["MIN_Q_W"]
        self.threshold = dic_traffic_env_conf["THRESHOLD"]
        
        if cnt_round == 0:

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.load_network("round_0_inter_{0}".format(intersection_id))
            else:
                self.q_network = self.build_q_network()
                self.a_network = self.build_a_network()
            self.q_network_bar = self.build_network_from_copy(self.q_network)
            self.a_network_bar = self.build_network_from_copy(self.a_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))

                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] *
                                    self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            cnt_round-1,
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                            cnt_round-1, self.intersection_id))
            except Exception:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    def cal_input_len(self):
        N = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        for feat_name in used_feature:
            if "num_in_seg" in feat_name:
                N += 12*4
            elif "new_phase" in feat_name:
                N += 0
            else:
                N += 12
        return N

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s_q.h5" % file_name))
        self.a_network = load_model(os.path.join(file_path, "%s_a.h5" % file_name))
        print("succeed in loading model %s" % file_name)

    def load_network_transfer(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_TRANSFER_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s_q.h5" % file_name))
        self.a_network = load_model(os.path.join(file_path, "%s_a.h5" % file_name))
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s_q_t.h5" % file_name))
        self.a_network_bar = load_model(os.path.join(file_path, "%s_a_t.h5" % file_name))
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_q.h5" % file_name))
        self.a_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_a.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_q_t.h5" % file_name))
        self.a_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_a_t.h5" % file_name))

    def build_network(self):
        raise NotImplementedError
       

    @staticmethod
    def build_memory():
        return []

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network

    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs, shuffle=False,
                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
