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
    env = make_env('Breakout-v0', 84, 84)
    n_iter = 613
    num_samples = 256
    batch_size = 32
    enc_threshold = 2.5
    init_enc_threshold = 1.0 
    use_encoder = 1
    if use_encoder:
        model_name = 'enc-mult-rew'
    else:
        model_name = 'no-enc'
    
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

    if use_encoder:
        encoder = Encoder(encoder_args)
        obs_encoded_shape = encoder.obs_encoded.get_shape().as_list()
        policy = PPO(policy_graph_args, adv_args, obs_encoded_shape)
        rnd = RND(obs_encoded_shape, rnd_args)
        agent = Agent(env, policy, encoder, rnd, replay_buffer, logger, agent_args)
    else:
        policy = PPO(policy_graph_args, adv_args, (None, 84, 84, 1))
        rnd = RND((None, 84, 84, 1), rnd_args)
        agent = NoEncoderAgent(env, policy, rnd, replay_buffer, logger, agent_args)
    agent.logger.model_name = model_name

    # training parameters
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
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
                print('starting training...')
                for itr in range(n_iter):
                    start = time.time()
                    agent.train(batch_size, num_samples, enc_threshold, itr)
                    end = time.time()
                    print('completed itr {} in {}sec...\r'.format(str(itr), int(end-start)))
                    print('size of logger:{}, size of buf:{}'.format(logger.size, len(replay_buffer.master_replay)))

            finally: # safe exit sooner
                if save:
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
                if act == 0:
                    if use_encoder:
                        obs = encoder.get_encoding([obs])
                    else:
                        obs = [obs]
                    act = policy.get_best_action(obs)
                elif act not in range(17):
                    if use_encoder:
                        obs = encoder.get_encoding([obs])
                    else:
                        obs = [obs]
                    act = policy.sample(obs)
                    # act = policy.get_best_action(enc_ob)
                    
                obs, rew, done, _ = env.step(act)
                if done: break

                
