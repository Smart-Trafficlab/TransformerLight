"""
This part is used to test the well-trianed models
"""
import json
import os
import time
from multiprocessing import Process
from utils import config
from utils.utils import merge
from utils.cityflow_env import CityFlowEnv
import argparse
import shutil

multi_process = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo",       type=str,               default='benchmark_TransformerLight')
    parser.add_argument("-old_memo",   type=str,               default='benchmark_TransformerLight')
    parser.add_argument("-model",       type=str,               default="General_TransformerLight") # AdvancedMPLight
    parser.add_argument("-old_dir",    type=str,               default='/home/wq/transformer_light/model/benchmark_TransformerLight/anon_3_4_jinan_real.json_01_25_11_50_03')
    parser.add_argument("-old_round",  type=str,                default="round_0")

    parser.add_argument("-workers",     type=int,                default=3)
    parser.add_argument("-hangzhou",    action="store_true",     default=False)
    parser.add_argument("-jinan",       action="store_true",     default=True)
    parser.add_argument("-newyork2", action="store_true",        default=True)
    return parser.parse_args()


def main(args):
    # preapre the data
    if args.hangzhou:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json",  "anon_4_4_hangzhou_real_5816.json"]
        num_rounds = 1
        template = "real-world/Hangzhou"
        
    elif args.jinan:

        count = 3600
        road_net = "3_4"
        traffic_file_list = [ #"anon_3_4_jinan_real_2000.json",
             "anon_3_4_jinan_real.json",
                            # "anon_3_4_jinan_real_2500.json"
             ]
        num_rounds = 1
        template = "real-world/Jinan"

    elif args.newyork2:
        count = 3600
        road_net = "28_7"
        traffic_file_list = [
            "anon_28_7_newyork_real_double.json"
            , "anon_28_7_newyork_real_triple.json"
            ]
        num_rounds = 1
        template = "real-world/newyork_28_7"

    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)

    old_memo = args.old_memo
    old_dir = args.old_dir
    old_round = args.old_round
    old_model_path = os.path.join(old_dir)

    process_list = []
    n_workers = args.workers

    for traffic_file in traffic_file_list:
        dic_agent_conf_extra = {
            "CNN_layers": [[32, 32]],
        }
        deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

        dic_traffic_env_conf_extra = {
            "MIN_Q_W": 0.005,
            "test_rounds":old_round,
            "MIN_ACTION_TIME": 15,
            "MEASURE_TIME": 15,
            "OBS_LENGTH": 167,  # 11*15
            "OBS_LENGTH_Q": 167,
            "IS_ACTOR_CRITIC": True,
            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": 1,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,
            "MODEL_NAME": args.model,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "LIST_STATE_FEATURE": [
                "new_phase",
                "lane_num_vehicle_in",
                "lane_num_vehicle_out",
                "lane_queue_vehicle_in",
                "lane_queue_vehicle_out",
                "traffic_movement_pressure_queue_efficient",
                "lane_run_in_part",
                "lane_queue_in_part",
                "num_in_seg",
            ],

            "DIC_REWARD_INFO": {
                "queue_length": -0.25,
            },
        }

        # change the model path to the old model path
        dic_path = {
            "PATH_TO_MODEL": old_model_path,  # use old model path
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, traffic_file + "_" +
                                                   time.strftime('%m_%d_%H_%M_%S', time.localtime(
                                                       time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net))
        }

        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

        multi_process = True
        if multi_process:
            tsr = Process(target=testor_wrapper,
                          args=(deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                dic_path,
                                old_round))
            process_list.append(tsr)
        else:
            testor_wrapper(deploy_dic_agent_conf,
                           deploy_dic_traffic_env_conf,
                           dic_path,
                           old_round)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return args.memo


def testor_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, old_round):
    testor = Testor(dic_agent_conf,
                    dic_traffic_env_conf,
                    dic_path,
                    old_round)
    testor.main(old_round)
    print("============= restor wrapper end =========")


class Testor:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, old_round):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self._path_check()
        self._copy_conf_file()
        self._copy_anon_file()

        agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
        # use one-model
        self.agent = config.DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(0)
        )

        self.path_to_log = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        self.env = CityFlowEnv(path_to_log=self.path_to_log,
                               path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                               dic_traffic_env_conf=self.dic_traffic_env_conf)
        # TODO, load the pretrained model
        self.agent.load_network("{0}_inter_0".format(old_round))


    def main(self, old_round):
        #rounds = ["round_" + str(i) for i in range(29, 30)]
    
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", old_round)
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        self.env = CityFlowEnv(path_to_log=self.path_to_log,
                                path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                                dic_traffic_env_conf=self.dic_traffic_env_conf)
        # TODO, load the pretrained model
        self.agent.load_network("{0}_inter_0".format(old_round))

        self.run()

    def run(self):
        done = False
        step_num = 0

        state = self.env.reset()
        running_start_time = time.time()
        while not done and step_num < int(self.dic_traffic_env_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            step_start_time = time.time()
            
        
            action_list = self.agent.choose_action( state)
            print("time: {0}, running_time: {1}".format(
                self.env.get_current_time() - self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                time.time() - step_start_time))


            next_state = self.env.step(action_list)

            # print("time: {0}, running_time: {1}".format(
            #     self.env.get_current_time() - self.dic_traffic_env_conf["MIN_ACTION_TIME"],
            #     time.time() - step_start_time))
            state = next_state
            step_num += 1

        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("=========== start env logging ===========")
        self.env.batch_log_2()
        log_time = time.time() - log_start_time
        # self.env.end_anon()
        print("running_time: ", running_time)
        print("log_time: ", log_time)

    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

    def _copy_conf_file(self, path=None):
        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["TRAFFIC_FILE"]),
                        os.path.join(path, self.dic_traffic_env_conf["TRAFFIC_FILE"]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_traffic_env_conf["ROADNET_FILE"]))


if __name__ == "__main__":
    args = parse_args()
    main(args)
