import tensorflow as tf
import numpy as np 
import pickle
from pathlib import Path
import gym 
import matplotlib
import time
import os

from utils import Logger, MasterBuffer, Network, make_env
from policy import PPO
from novelty import Encoder, RND
from dynamics import DynamicsModel
from agent import Agent
from noEncAgent import NoEncoderAgent
from args import get_args 


if __name__ == '__main__':
    env = make_env('MontezumaRevenge-v0', 84, 84)
    n_iter = 500
    num_samples = 5000
    batch_size = 32
    enc_threshold = 1.7
    init_enc_threshold = 1.2
    use_encoder = 1
    if use_encoder:
        model_name = 'enc-init-policy-base-mr'
    else:
        model_name = 'no-enc-base-mr'
    
    train = 1
    restore = 0
    save = 1
    
    test_run = 1
    view = 0
    
    if view:
        train = False
        restore = True
        save = False
    if test_run:
        num_samples = 10
        n_iter = 5
        batch_size = 8
        save = False
        restore = False

    policy_graph_args, adv_args, encoder_args, rnd_args, agent_args = get_args(env, test_run=test_run)
    replay_buffer = MasterBuffer(max_size=5000)
    logger = Logger(max_size=100000)

    if use_encoder: agent_args['use_encoder'] = 1
    else: agent_args['use_encoder'] = 0

    policy = PPO(policy_graph_args, adv_args, (None, 84, 84, 1))
    rnd = RND((None, 84, 84, 1), rnd_args)
    agent = Agent(env, policy, rnd, replay_buffer, logger, agent_args)
    agent.logger.model_name = model_name

    # training parameters
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        writer = tf.summary.FileWriter('./tf_logs')
        sess.run(tf.global_variables_initializer())
        agent.set_session(sess)
        if restore: 
            saver.restore(sess, "./model_data/model-{}.ckpt".format(model_name))
            agent.num_random_samples = 5

        if train:
            try:
                if use_encoder:
                    print('training encoder...')
                    agent.init_encoder(batch_size, num_samples, init_enc_threshold)
                
                print('initializing obs running mean...')
                agent.init_obsmean()
                
                print('starting training...')
                for itr in range(n_iter):
                    start = time.time()
                    agent.train(batch_size, num_samples, enc_threshold, itr, writer)
                    end = time.time()
                    print('completed itr {} in {}sec...\r'.format(str(itr), int(end-start)))

            finally: # safe exit sooner
                if save:
                    writer.close()
                    logger.export()
                    saver.save(sess, "./model_data/model-{}.ckpt".format(model_name))
        
        if view: # view
            obs = env.reset()
            while True:
                env.render()
                try:
                    act = int(input('Press a key to continue...'))
                except (ValueError, TypeError):
                    act = 20

                if act == 0:                # best action on 0
                    obs = [obs]
                    act = policy.get_best_action(obs)
                elif act not in range(17):  # sample on ENTER
                    obs = [obs]
                    act = policy.sample(obs)
                    
                obs, rew, done, _ = env.step(act)
                if done: break

                
