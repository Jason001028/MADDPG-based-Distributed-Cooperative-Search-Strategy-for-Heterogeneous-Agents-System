# #! /usr/bin/env python
import random
import torch
import time
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from arguments import Args as args
from core.logger import Logger
from core.actor import actor_worker
from core.evaluator import evaluate_worker
from core.learner import learn

# set logging level
logger = Logger(logger="dual_arm_multiprocess")

# #随机生成障碍
# origin_obstacle_states = []
# for _ in range(60):
#     x = random.randint(1,19)
#     y = random.randint(1,19)
#     if [x,y] not in origin_obstacle_states:
#         origin_obstacle_states.append([x,y])
# # save obs
# with open('origin_obstacle_states.txt', 'w') as f:
#     for item in origin_obstacle_states:
#         f.write("%s\n" % item)
###加载地图模型文件,16*16情形
L1 = pd.read_excel("D:\origin_obstacle_states_mid.xlsx",engine="openpyxl",sheet_name="Sheet1")
origin_obstacle_states = []
i=0
for i in range(0, 16):
    j = 0
    for j in range(0, 16):
      x = i
      y = L1.iat[j, i]
      if (y != 0):
        origin_obstacle_states.append([x,y])





# ###加载地图模型文件,25*25情形
# L1 = pd.read_excel("D:\origin_obstacle_states_verylarge.xlsx",engine="openpyxl",sheet_name="Sheet1")
# origin_obstacle_states = []
# i=0
# for i in range(0, 16):
#     j = 0
#     for j in range(0, 16):
#       x = i
#       y = L1.iat[j, i]
#       if (y != 0):
#         origin_obstacle_states.append([x,y])

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)




def train():#代码框架的核心
    train_params = args.train_params
    env_params = args.env_params
    actor_num = train_params.actor_num
    model_path = os.path.join(train_params.save_dir, train_params.env_name)

    if not os.path.exists(train_params.save_dir):
        os.mkdir(train_params.save_dir)
        logger.info(f'creating directory {train_params.save_dir}')
    else:
        logger.info(f'directory {train_params.save_dir} already exists')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        logger.info(f'creating directory {model_path}')
    else:
        logger.info(f'directory {model_path} already exists')
    setup_seed(args.seed)
    logger.info(f'New experiment date: {args.date}, seed: {args.seed}')
    # starting multiprocess
    ctx = mp.get_context("spawn") # using shared cuda tensor should use 'spawn'
    # queue to transport data
    data_queue = ctx.Queue()
    evalue_queue = ctx.Queue()
    actor_queues = [ctx.Queue() for _ in range(actor_num)]
    logger.info("Starting learner process...")

    # starting actor worker process
    logger.info("Starting actor process...")
    actor_processes = []
    for i in range(actor_num):
        actor = ctx.Process(
            target = actor_worker,
            args = (
                data_queue,
                actor_queues[i],
                i,
                logger,
                origin_obstacle_states
            )
        )
        logger.info(f"Starting actor:{i} process...")
        actor.start()
        actor_processes.append(actor)
        time.sleep(1)

    logger.info(f"starting learner worker process...")
    learner_process = ctx.Process(
        target = learn,
        args = (
            model_path,
            data_queue,
            evalue_queue,
            actor_queues,
            logger,
        )
    )
    logger.info(f"Starting evaluate process...")
    evaluate_process = ctx.Process(
        target = evaluate_worker,
        args = (
            train_params,
            env_params,
            model_path,
            train_params.evalue_time,
            evalue_queue,
            logger,
            origin_obstacle_states
        )
    )
    learner_process.start()
    time.sleep(2)
    evaluate_process.start()
    logger.info(f"actor_process:{actor_processes}")
    logger.info(f"evaluate_process:{evaluate_process}")
    logger.info(f"learner_process:{learner_process}")
    learner_process.join()
    time.sleep(100000)

if __name__ == "__main__":
    # set threading num to avoid possess too much cpu resource
    import os
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"
    torch.set_num_threads(1)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # necessary for a larger shared memory buffer
    mp.set_sharing_strategy('file_system')

    # df = pd.read_excel('D:\研一大冤种\七月大工程2.0\smog_project_heterogeneous\saved_models/agent_cover.xlsx', sheet_name='Sheet1')
    # df.drop(df.index, inplace=True)
    # df.to_excel('D:\研一大冤种\七月大工程2.0\smog_project_heterogeneous\saved_models/agent_cover.xlsx', index=False)

    train()