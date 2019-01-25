import tensorflow as tf
import numpy as np

class Agent(object):
    def __init__(self, env, policy, encoder, rnd, replay_buffer, logger, args):
        self.env = env
        # Models
        self.policy = policy
        self.encoder = encoder
        self.rnd = rnd
        # self.dynamics = dynamics
        # Utils
        self.replay_buffer = replay_buffer
        self.logger = logger
        # Args
        self.rnd_train_itr = args['rnd_train_itr']
        self.num_actions_taken_conseq = args['num_actions_taken_conseq']
        self.num_random_samples = args['num_random_samples']
        self.algorithm_rollout_rate = args['algorithm_rollout_rate']
        self.log_rate = args['log_rate']
        self.p_rand_act = args['p_rand']
        self.max_ran_samples = args['max_ran_samples']
        
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)
        self.encoder.set_sess(sess)
        self.rnd.set_sess(sess)
        # self.dynamics.set_session(sess)
        
    def sample_env(self, batch_size, num_samples, shuffle, action_selection):
        obs = self.env.reset()
        for _ in range(num_samples):
            # inject random samples into samples
            if action_selection == 'random':
                act = self.env.action_space.sample()
            else:  # action_selection == algorithm 
                enc_ob = self.encoder.multi_t_resize([obs])
                act = self.policy.get_best_action(enc_ob)
            n_ob, rew, done, _ = self.env.step(act)
            self.replay_buffer.record(obs, act, rew, n_ob, done)
            obs = n_ob if not done else self.env.reset()
        
        # sync logger work
        obs_n, act_n, n_obs_n = self.replay_buffer.get_obs_act_nobs()
        resized_obs_n = self.encoder.multi_t_resize(obs_n)
        resized_n_obs_n = self.encoder.multi_t_resize(n_obs_n)

        logprobs = self.policy.get_logprob(resized_obs_n, act_n)
        self.replay_buffer.set_logprobs(logprobs)
        self.replay_buffer.temp_replay.obs = list(resized_obs_n)
        self.replay_buffer.temp_replay.nxt_obs = list(resized_n_obs_n)
        self.replay_buffer.merge_temp()
        return self.replay_buffer.get_all(batch_size, shuffle=shuffle)
        
    def get_data(self, batch_size, num_samples, itr):
        if itr < self.num_random_samples or (np.random.uniform() <= self.p_rand_act and itr < self.max_ran_samples):
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
            # # update dynamics 
            # for _ in range(self.dynamics_train_itr):
            #     # encode states
            #     reshaped_acts = acts.reshape(acts.shape[0], 1)
            #     loss = self.dynamics.update(enc_obs, reshaped_acts, enc_n_obs)
            #     self.logger.log('dynamics', ['loss'], [loss])

            for _ in range(self.rnd_train_itr):
                rnd_loss = self.rnd.train(obs)

            total_rewards = self.rnd.modify_rewards(obs, rewards)
            critic_loss = self.policy.train_critic(obs, n_obs, total_rewards, dones)
            adv = self.policy.estimate_adv(obs, total_rewards, n_obs, dones)
            actor_loss = self.policy.train_actor(obs, acts, logprobs, adv)
           
            if itr % self.log_rate:
                self.logger.log('density', ['loss'], [rnd_loss])
                self.logger.log('policy', ['actor_loss', 'critic_loss'], [actor_loss, critic_loss])
