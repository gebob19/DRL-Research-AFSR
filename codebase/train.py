import tensorflow as tf
import numpy as np 
import pickle
from pathlib import Path
import gym 
import matplotlib
import time
import os

from utils import Logger, MasterBuffer, Network
from policy import Policy
from density import DensityModel
from dynamics import DynamicsModel

class Agent(object):
    def __init__(self, env, policy, density, dynamics, replay_buffer, logger, args):
        self.env = env
        # Models
        self.policy = policy
        self.density = density
        self.dynamics = dynamics
        # Utils
        self.replay_buffer = replay_buffer
        self.logger = logger
        # Args
        self.density_train_itr = args['density_train_itr']
        self.dynamics_train_itr = args['dynamics_train_itr']
        self.num_actions_taken_conseq = args['num_actions_taken_conseq']
        self.exploitation_threshold = args['exploitation_threshold']
        self.num_random_samples = args['num_random_samples']
        self.algorithm_rollout_rate = args['algorithm_rollout_rate']
        self.profit_fcn = self.density.get_bonus
        
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)
        self.density.set_session(sess)
        self.dynamics.set_session(sess)
        
    def sample_env(self, batch_size, num_samples, shuffle, action_selection):
        obs = self.env.reset()
        taking_action = False
        
        for _ in range(num_samples):
            if action_selection == 'random':
                act = self.env.action_space.sample()
            elif action_selection == 'algorithm':
                if not taking_action:
                    actions = self.get_action(obs, self.num_actions_taken_conseq)
                    action_index, taking_action = 0, True
                act = actions[action_index][0]
                action_index += 1
                # check for end of actions
                if action_index == len(actions): taking_action = False
            # execute action and record
            n_ob, rew, done, _ = env.step(act)
            self.replay_buffer.record(obs, act, rew, n_ob, done)
            obs = n_ob if not done else env.reset()
        
        logger.log('env', ['rewards'], [np.sum(self.replay_buffer.get_temp_rewards())])
        # get logprobs of taking actions w.r.t current policy
        obs, actions = self.replay_buffer.get_actions()
        logprobs = sess.run(policy.logprob, feed_dict={
            policy.obs: obs,
            policy.act: actions
        })
        self.replay_buffer.set_logprobs(logprobs)
        self.replay_buffer.merge_temp()
        return self.replay_buffer.get_all(batch_size, shuffle=shuffle)
        
    def get_action(self, obs, num_actions):
        # Encode observation from density encoder
        enc_obs = self.density.get_encoding([obs])[0]
        # get action from state dynamics
        actions, profit = self.dynamics.get_best_actions(enc_obs, self.profit_fcn, num_actions)

        # To exploit or not to exploit, is the question...
        if profit < self.exploitation_threshold:  
            actions = [[policy.get_best_action(obs)]]
        else:
            actions = actions.eval()

        logger.log('dynamics', ['max_profit'], [profit])
        return actions
        
    def get_data(self, batch_size, num_samples, itr):
        if itr < self.num_random_samples:
            return self.sample_env(batch_size, num_samples, shuffle=True, action_selection='random')
        
        if itr % self.algorithm_rollout_rate == 0:
            return self.sample_env(batch_size, num_samples, shuffle=True, action_selection='algorithm')
        else:
            return self.replay_buffer.get_all(batch_size, master=True, shuffle=True, size=num_samples)
    
    def train(self, batch_size, num_samples, itr):
        obsList, actsList, rewardsList, n_obsList, donesList, logprobsList = self.get_data(batch_size, num_samples, itr)
        self.replay_buffer.flush_temp()
        # process all data in batches 
        for obs, acts, rewards, n_obs, dones, logprobs in zip(obsList, actsList, rewardsList, n_obsList, donesList, logprobsList):
            # train density model
            for _ in range(self.density_train_itr):
                s1, s2, target = self.replay_buffer.get_density_batch(obs, batch_size)
                ll, kl, elbo = self.density.update(s1, s2, target)
                self.logger.log('density', ['logloss', 'kl', 'elbo'], [ll, kl, elbo])
            
            # update dynamics 
            for _ in range(self.dynamics_train_itr):
                # encode states
                enc_obs = self.density.get_encoding(obs)
                enc_n_obs = self.density.get_encoding(n_obs)
                reshaped_acts = acts.reshape(acts.shape[0], 1)
                loss = self.dynamics.update(enc_obs, reshaped_acts, enc_n_obs)
                self.logger.log('dynamics', ['loss'], [loss])

            # train critic & actor
            critic_loss = self.policy.train_critic(obs, n_obs, rewards, dones)
            adv = self.policy.estimate_adv(obs, rewards, n_obs, dones)
            actor_loss = self.policy.train_actor(obs, acts, logprobs, adv)
            self.logger.log('policy', ['actor_loss', 'critic_loss'], [actor_loss, critic_loss])

if __name__ == '__main__':
    #  lets get ready to rumble
    env = gym.make('MontezumaRevenge-v0')
    
    policy_graph_args = {
        'ob_dim': env.observation_space.shape,
        'act_dim': env.action_space.n,
        'clip_range': 0.2,
        'conv_depth': 5,
        'filter_size': 32,
        'learning_rate': 5e-3,
        'num_target_updates': 10,
        'num_grad_steps_per_target_update': 10
    }
    adv_args = {
        'gamma': 0.9999999
    }

    density_graph_args = {
        'ob_dim': env.observation_space.shape,
        'learning_rate': 5e-3,
        'z_size': 32,
        'kl_weight': 1e-2,
        'conv_depth': 5,
        'hid_size': 32,
        'n_hidden': 2,
        'bonus_multiplier': 1
    }

    action_low = 0
    action_high = env.action_space.n
    dynamics_graph_args = {
        'enc_dim': density_graph_args['z_size'],
        'act_dim': 1,
        'action_low': action_low,
        'action_high': action_high,
        'learning_rate': 1e-3,
        'hid_size': 64,
        'n_hidden': 2 
    }
    dynamics_rollout_args = {
        'horizon': 30,
        'num_rollouts': 200,
    }
    agent_args = {
        'density_train_itr': 100,
        'dynamics_train_itr': 10,
        'num_actions_taken_conseq': 10,
        # hyperparameterized
        'exploitation_threshold': 0,
        'num_random_samples': 100,
        # very expensive to rollout so we set a rate
        'algorithm_rollout_rate': 5
    }

    replay_buffer = MasterBuffer(max_size=30000)
    logger = Logger()

    policy = Policy(policy_graph_args, adv_args)
    dynamics = DynamicsModel(dynamics_graph_args, dynamics_rollout_args)
    density = DensityModel(density_graph_args)
    
    agent = Agent(env, policy, density, dynamics, replay_buffer, logger, agent_args)

    # training parameters
    exploitations_to_test = [np.random.randint(50, 100)]
    n_iter = 51
    num_samples = 100
    batch_size = 32
    n_export = 10
    
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "./model_data/model-1547614855.98.ckpt")

        for exp in exploitations_to_test:
            agent.exploitation_threshold = exp
            agent.set_session(sess)

            print('testing with exploitation: {}'.format(exp))
            # start training
            for itr in range(n_iter):
                start = time.time()
                agent.train(batch_size, num_samples, itr)
                end = time.time()
                print('completed itr {} in {}sec...\r'.format(str(itr), int(end-start)))
                
                if itr % n_export == 0 and itr != 0:
                    logger.export()
                    print('Exported logs...')
                    saver.save(sess, './model_data/model-first.ckpt')
