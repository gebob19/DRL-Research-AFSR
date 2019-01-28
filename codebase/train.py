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
    n_iter = 100
    num_samples = 100
    batch_size = 32
    
    train = True
    restore = False
    save = True
    
    test_run = True

    if test_run:
        num_samples = 10
        n_iter = 5
        batch_size = 8
        save = False
        restore = False

    policy_graph_args, adv_args, encoder_args, rnd_args, agent_args = get_args(env, test_run=test_run)
    replay_buffer = MasterBuffer(max_size=3000)
    logger = Logger(max_size=100000)

    encoder = Encoder(encoder_args)
    obs_encoded_shape = encoder.obs_encoded.get_shape().as_list()
    policy = PPO(policy_graph_args, adv_args, obs_encoded_shape)
    rnd = RND(obs_encoded_shape, rnd_args)

    agent = Agent(env, policy, encoder, rnd, replay_buffer, logger, agent_args)

    # training parameters
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        agent.set_session(sess)
        if restore: 
            saver.restore(sess, "./model_data/model.ckpt")
            agent.num_random_samples = 5

        if train:
            try:
                print('Starting training...')
                for itr in range(n_iter):
                    start = time.time()
                    agent.train(batch_size, num_samples, itr)
                    end = time.time()
                    print('completed itr {} in {}sec...\r'.format(str(itr), int(end-start)))
                    print('size of logger:{}, size of buf:{}'.format(logger.size, len(replay_buffer.master_replay)))
            finally: # safe exit sooner
                if save:
                    logger.export()
                    saver.save(sess, './model_data/model.ckpt')
        else: # view
            obs = env.reset()
            while True:
                env.render()
                try:
                    act = int(input('Press a key to continue...'))
                except (ValueError, TypeError):
                    act = 20
                if act not in range(17):
                    enc_ob = encoder.get_encoding([obs])
                    act = policy.get_best_action(enc_ob)
                obs, rew, done, _ = env.step(act)
                if done: break

                
