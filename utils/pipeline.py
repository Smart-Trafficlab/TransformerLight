from .generator import Generator
from . import model_test
import json
import shutil
import os
import time
from multiprocessing import Process
import pickle


def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


def generator_wrapper(cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf, memory):
    generator = Generator(cnt_round=cnt_round,
                          cnt_gen=cnt_gen,
                          dic_path=dic_path,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          )
    print("make generator")
    generator.train_model(memory)
    print("generator_wrapper end")
    return

def generator_wrapper2(cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):
    generator = Generator(cnt_round=cnt_round,
                          cnt_gen=cnt_gen,
                          dic_path=dic_path,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          )
    print("make generator")
    generator.load_data()
    print("generator_wrapper end")
    return


class Pipeline:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.initialize()
        
    def load_data(self):
        path1 = self.dic_path["PATH_TO_MEMORY"]
        print(path1)

        with open(path1, "rb") as f:
            memory = pickle.load(f)

        return memory

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

    def run(self):
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()
        
        replay_memory = self.load_data()
        
        for cnt_round in range(self.dic_traffic_env_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            print("=============== update model =============")
            generator_start_time = time.time()
            for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                generator_wrapper(cnt_round=cnt_round,
                                  cnt_gen=cnt_gen,
                                  dic_path=self.dic_path,
                                  dic_agent_conf=self.dic_agent_conf,
                                  dic_traffic_env_conf=self.dic_traffic_env_conf,
                                  memory=replay_memory)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time


            print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            if cnt_round + 0 >= self.dic_traffic_env_conf["NUM_ROUNDS"]:
                pass
            model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round,
                                self.dic_traffic_env_conf["RUN_COUNTS"], self.dic_traffic_env_conf)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time


            print("Generator time: ", generator_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
            f_time.write("{0}\t{1}\t{2}\n".format(generator_total_time, test_evaluation_total_time,
                                                  time.time()-round_start_time))
            f_time.close()
