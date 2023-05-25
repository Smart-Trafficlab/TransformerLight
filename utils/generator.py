from .config import DIC_AGENTS
from .cityflow_env import CityFlowEnv
import time
import os
import copy
import pickle
from tensorflow.keras.callbacks import EarlyStopping


class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    intersection_id=str(i)
                )
                self.agents[i] = agent
            print("Create intersection agent time: ", time.time()-start_time)

        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )

    def load_data(self):
        path1 = self.dic_path["PATH_TO_MEMORY"]
        print(path1)

        with open(path1, "rb") as f:
            memory = pickle.load(f)

        return memory

    def train_model(self, memory):
        # memory = self.load_data()

        self.agents[0].train_network(memory)

        # Xs, Y = self.agent[0].prepare_Xs_Y(memory)

        # epochs = self.dic_agent_conf["EPOCHS"]
        # batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))
        #
        # early_stopping = EarlyStopping( monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        #
        # self.agent[0].q_network.fit(Xs, Y, batch_size=batch_size, epochs=epochs, shuffle=True,
        #                             verbose=2, validation_split=0.3, callbacks=[early_stopping])

        self.agents[0].save_network("round_{0}_inter_{1}".format(self.cnt_round, 0))
        
        if 'IS_ACTOR_CRITIC' in self.dic_traffic_env_conf:
            self.agents[0].save_network_bar("round_{0}_inter_{1}".format(self.cnt_round, 0))
        print("=============  save model  finished ==============")

