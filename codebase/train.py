import tensorflow as tf
import numpy as np 
import pickle
from pathlib import Path
import gym 
import matplotlib
import time
import os

from utils import Logger, MasterBuffer, Network
from policy import PPO
from novelty import Encoder, RND
from dynamics import DynamicsModel
from agent import Agent
from args import get_args 

if __name__ == '__main__':
    env = gym.make('MontezumaRevenge-v0')
    policy_graph_args, adv_args, rnd_args, agent_args = get_args(env)
    replay_buffer = MasterBuffer(max_size=30000)
    logger = Logger()

    encoder = Encoder(150, 'imagenet', True, 'Encoder-TargetNetwork')
    policy = PPO(policy_graph_args, adv_args, encoder.x, encoder.encoded_tensor)
    rnd = RND(encoder, 60, rnd_args)
    # TODO: Multithreaded image resizing to (224, 224, 3)
    agent = Agent(env, policy, encoder, rnd, replay_buffer, logger, agent_args)
    dynamics = DynamicsModel(dynamics_graph_args, dynamics_rollout_args)
    # training parameters
    exploitations_to_test = [np.random.randint(50, 100)]
    n_iter = 1
    num_samples = 5
    batch_size = 2
    n_export = 10
    train = True
    
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        agent.set_session(sess)
        if train:
            # saver.restore(sess, "./model_data/model-1547614855.98.ckpt")
            # start training
            print('Starting traininng...')
            for itr in range(n_iter):
                start = time.time()
                agent.train(batch_size, num_samples, itr)
                end = time.time()
                print('completed itr {} in {}sec...\r'.format(str(itr), int(end-start)))
                
                if itr % n_export == 0 and itr != 0:
                    logger.export()
                    print('Exported logs...')
            saver.save(sess, './model_data/model-{}.ckpt'.format(time.time()))
        # else: # view
        #     # saver.restore(sess, "./model_data/model-first.ckpt")
        #     obs = env.reset()
        #     while True:
        #         env.render()
        #         actions, profit, policy_stats = agent.get_action(obs, 1, debug=True)
        #         act = actions[0][0]
        #         print('Action:{}, Pred_Sum_Profit:{}, Act_Distrib:{}'.format(
        #             act, profit, policy_stats
        #         ))
        #         n_ob, rew, done, _ = env.step(act)
        #         input('Press a key to continue...')
        #         if done: break

                
