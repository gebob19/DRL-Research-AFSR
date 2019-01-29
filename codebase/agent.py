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
        self.encoder_train_itr = args['encoder_train_itr']

        self.num_conseq_rand_act = args['num_conseq_rand_act']
        self.num_random_samples = args['num_random_samples']
        self.algorithm_rollout_rate = args['algorithm_rollout_rate']
        self.log_rate = args['log_rate']
        self.p_rand_act = args['p_rand']
        
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)
        self.encoder.set_sess(sess)
        self.rnd.set_sess(sess)
        # self.dynamics.set_session(sess)
        
    def sample_env(self, batch_size, num_samples, shuffle, action_selection):
        obs = self.env.reset()
        done, i = False, 0
        takingRandom = False
        n_lives = 6
        num_rand = 0
        ignore = 0
        while i < num_samples or (not done and i < (100 + num_samples)):
            if num_rand == self.num_conseq_rand_act: takingRandom = False
            # inject random samples into samples
            if action_selection == 'random': #or np.random.uniform() <= self.p_rand_act or takingRandom:
                act = self.env.action_space.sample()
                # if not takingRandom: 
                #     takingRandom = True
                #     num_rand = 1
            else:  # action_selection == algorithm 
                enc_ob = self.encoder.get_encoding([obs])
                act = self.policy.sample(enc_ob)
            n_ob, rew, done, info = self.env.step(act)

            # dont record when agent dies
            if info['ale.lives'] != n_lives:
                ignore = 18
                n_lives -= 1
            if not done:
                if ignore > 0:
                    ignore -= 1
                else:
                    self.replay_buffer.record(obs, act, rew, n_ob, done) # temp solution for enc req.
                    obs = n_ob
                    i += 1
            else:
                obs = self.env.reset()
                done = True
                i += 1
        
        # sync logger work
        obs_n, act_n, _ = self.replay_buffer.get_obs_act_nobs()
        enc_obs_n = self.encoder.get_encoding(obs_n)
        logprobs = self.policy.get_logprob(enc_obs_n, act_n)

        self.replay_buffer.set_logprobs(logprobs)
        self.replay_buffer.merge_temp()
        return self.replay_buffer.get_all(batch_size, shuffle=shuffle)
        
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
            enc_obs = self.encoder.get_encoding(obs)
            enc_n_obs = self.encoder.get_encoding(n_obs)

            for _ in range(self.encoder_train_itr):
                enc_loss = self.encoder.train(enc_obs, enc_n_obs, acts)

            for _ in range(self.rnd_train_itr):
                rnd_loss = self.rnd.train(enc_obs)

            total_rewards = self.rnd.modify_rewards(enc_obs, rewards)
            critic_loss = self.policy.train_critic(enc_obs, enc_n_obs, total_rewards, dones)
            adv = self.policy.estimate_adv(enc_obs, total_rewards, enc_n_obs, dones)
            actor_loss = self.policy.train_actor(enc_obs, acts, logprobs, adv)
           
            if itr % self.log_rate == 0:
                self.logger.log('density', ['loss'], [rnd_loss])
                self.logger.log('policy', ['actor_loss', 'critic_loss'], [actor_loss, critic_loss])
                self.logger.log('encoder', ['loss'], [enc_loss])

    def enc_shuffle(self, o, a):
        indxs = np.arange(o.shape[0])
        np.random.shuffle(indxs)
        return o[indxs], a[indxs]

    def shuffle(self, o, a, r, n, d, l):
        indxs = np.arange(o.shape[0])
        np.random.shuffle(indxs)
        return o[indxs], a[indxs], r[indxs], n[indxs], d[indxs], l[indxs]