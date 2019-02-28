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
from args import get_args 


if __name__ == '__main__':
    env = make_env('MontezumaRevenge-v0', 84, 84)
    
    # main training params
    n_iter = 500
    num_samples = 8000
    batch_size = 64
    
    # encoder params
    use_encoder = 1
    enc_threshold = 1.9
    init_enc_threshold = 1.5
    if use_encoder:
        model_name = 'enc-base-mr-encthresholds_{}-{}-2'.format(init_enc_threshold, enc_threshold)
    else:
        model_name = 'no-enc-base-mr-2'
    
    train = 1
    restore = 0
    save = 1
    
    record = 0
    test_run = 0
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
    if record:
        train, save = False, False
        restore = True
        n_iter = 100
        
    if save or restore  or test_run:
        directory = "./model_{}".format(model_name)
        if not os.path.exists(directory):
            os.mkdir(directory)

    # setup graph 
    policy_graph_args, adv_args, rnd_args, agent_args = get_args(env, test_run=test_run)
    replay_buffer = MasterBuffer(max_size=5000)
    logger = Logger(max_size=100000)

    if use_encoder: agent_args['use_encoder'] = 1
    else: agent_args['use_encoder'] = 0

    policy = PPO(policy_graph_args, adv_args, (None, 84, 84, 1))
    rnd = RND((None, 84, 84, 1), rnd_args)
    agent = Agent(env, policy, rnd, replay_buffer, logger, agent_args)
    agent.logger.model_name = model_name

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        # gradient logs 
        writer = tf.summary.FileWriter('./tf_logs_{}'.format(model_name))
        sess.run(tf.global_variables_initializer())
        agent.set_session(sess)

        if restore: 
            saver.restore(sess, "./algo_data/model_{}/model.ckpt".format(model_name))
            agent.num_random_samples = 5

        # record frames of best mean & max rollouts
        if record:
            print('Starting to record...')
            br_frames, i, br_rew = [], 0, 0
            agent.init_obsmean()
            for itr in range(n_iter):
                int_rew, ext_rew, frames = agent.record(num_samples)
                agent.logger.log('env', ['int_rewards', 'ext_rewards'], [int_rew, ext_rew])
                mean_rollout = np.mean(int_rew) + np.mean(ext_rew)
                if mean_rollout > br_rew:
                    br_frames, i, br_rew = frames, itr, mean_rollout
            logger.log('env', ['frames'], [br_frames])
            logger.export()

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
                    saver.save(sess, "./model_{}/model.ckpt".format(model_name))
        
        if view: 
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

        

                
