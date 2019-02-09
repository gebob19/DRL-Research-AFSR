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
        self.last_enc_loss = None
        self.train_enc_next_itr = False
        # Args
        self.rnd_train_itr = args['rnd_train_itr']
        self.encoder_train_itr = args['encoder_train_itr']
        self.encoder_update_freq = args['encoder_update_freq']
        self.encoder_updates = args['encoder_updates']

        self.num_conseq_rand_act = args['num_conseq_rand_act']
        self.num_random_samples = args['num_random_samples']
        self.algorithm_rollout_rate = args['algorithm_rollout_rate']
        self.log_rate = args['log_rate']
        self.p_rand_act = args['p_rand']
        
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)
        # self.encoder.set_sess(sess)
        self.rnd.set_sess(sess)
        # self.dynamics.set_session(sess)

    def batch(self, eo, a, er, ir, en, d, batch_size, shuffle=True):
        if shuffle:
            indxs = np.arange(len(eo))
            np.random.shuffle(indxs)
            eo, a, er, ir, en, d = np.array(eo)[indxs], \
                np.array(a)[indxs], np.array(er)[indxs], np.array(ir)[indxs], \
                np.array(en)[indxs], np.array(d)[indxs]
        
        # batch up data
        batched_dsets = []
        for dset in [eo, a, er, ir, en, d]:
            bdset = []
            for i in range(0, len(dset), batch_size):
                 bdset.append(np.array(dset[i:i+batch_size]))
            batched_dsets.append(np.array(bdset))
        return tuple(batched_dsets)
        
    def sample_env(self, batch_size, num_samples, shuffle, algorithm='algorithm'):
        done, i = False, 0
        # n_lives, ignore = 6, 0
        obs_n, act_n, ext_rew_n, int_rew_n, n_obs_n, dones_n = [], [], [], [], [], []
        
        # policy rollout
        obs = self.env.reset()
        while i < num_samples or (not done and i < (100 + num_samples)):
            # enc_ob = self.encoder.get_encoding([obs])
            if algorithm == 'algorithm':
                act = self.policy.sample([obs])
            else: # algorithm == 'random'
                act = self.env.action_space.sample()

            n_obs, rew, done, info = self.env.step(act)
            int_rew = self.rnd.get_rewards([obs])[0]
            # dont record when agent dies
            # if info['ale.lives'] != n_lives:
            #     ignore = 18; n_lives -= 1
            # if ignore > 0: ignore -= 1
            # else:
            obs_n.append(obs); ext_rew_n.append(rew); n_obs_n.append(n_obs)
            act_n.append(act); dones_n.append(done); int_rew_n.append(int_rew)
            if done:
                obs = self.env.reset()
                done = True
                # n_lives, ignore = 6, 0
            i += 1

        # enc_obs = self.encoder.get_encoding(obs_n)
        # enc_n_obs = self.encoder.get_encoding(n_obs_n)
        # ext_rew_n = np.clip(ext_rew_n, -1, 1)

        obs, n_obs = self.norm(obs_n), self.norm(n_obs_n)
        int_rew_n = self.norm(int_rew_n) * 10
        ext_rew_n = np.array(ext_rew_n)

        self.logger.log('env', ['int_rewards', 'ext_rewards'], [int_rew_n, ext_rew_n])
        return self.batch(obs, act_n, ext_rew_n, int_rew_n, n_obs, dones_n, batch_size, shuffle)
        
        # sync logger work
        # obs_n, act_n, rew_n = self.replay_buffer.get_logger_work()
        # enc_obs_n = self.encoder.get_encoding(obs_n)
        # logprobs = self.policy.get_logprob(enc_obs_n, act_n)
        # self.replay_buffer.set_logprobs(logprobs)
        # self.replay_buffer.merge_temp()
        # return self.replay_buffer.get_all(batch_size, shuffle=shuffle)
    
    def norm_clip(self, obs):
        # normalize and clip before training
        nrm_obs = (obs - np.mean(obs)) / (np.var(np.array(obs)) + 1e-6)
        return np.clip(nrm_obs, -5, 5)
    
    def norm(self, r):
        return (r - np.mean(r)) / (np.var(np.array(r)) + 1e-6)
        
    def get_data(self, batch_size, num_samples, itr):
        # if itr < self.num_random_samples:
        #     return self.sample_env(batch_size, num_samples, shuffle=True, action_selection='random')
        # if itr % self.algorithm_rollout_rate == 0:
        return self.sample_env(batch_size, num_samples, shuffle=True)
        # else:
        #     return self.replay_buffer.get_all(batch_size, master=True, shuffle=True, size=num_samples)

    def init_encoder(self, batch_size, num_samples, loss_threshold):
        threshold_met, i = False, 0
        while not threshold_met and i < 1000:
            enc_obs, act_n, _, _, enc_n_obs, _  = self.sample_env(batch_size, num_samples, shuffle=True, algorithm='random')
            for b_eobs, b_acts, b_enobs in zip(enc_obs, act_n, enc_n_obs):
                for _ in range(self.encoder_train_itr):
                    enc_loss = self.policy.train_acthead(b_eobs, b_enobs, b_acts)
                    self.logger.log('encoder', ['loss'], [np.mean(enc_loss)])
                    if np.mean(enc_loss) < loss_threshold: threshold_met = True
                    i += 1
        if threshold_met: print('Encoder init threshold was met...')
        else: print('Encoder init threshold was NOT met...')
    
    def train(self, batch_size, num_samples, encoder_loss_thresh, itr):
        enc_obs, act_n, ext_rew_n, int_rew, enc_n_obs, dones_n = self.get_data(batch_size, num_samples, itr)
        for b_eobs, b_acts, b_erew, b_irew, b_enobs, b_dones in zip(enc_obs, act_n, ext_rew_n, int_rew, enc_n_obs, dones_n):
            
            # if self.train_enc_next_itr:
            #     for _ in range(self.encoder_train_itr):
            #         enc_loss = self.encoder.train(b_eobs, b_enobs, b_acts)
            #         self.logger.log('encoder', ['loss'], [np.mean(enc_loss)])

            rnd_loss = self.rnd.train(b_eobs)

            total_r = b_erew + b_irew
            # 1 critic temp soln
            critic_loss = self.policy.train_critic(b_eobs, b_enobs, total_r, b_dones)
            adv = self.policy.estimate_adv(b_eobs, total_r, b_enobs, b_dones)
            actor_loss = self.policy.train_actor(b_eobs, b_acts, adv)
            # if itr < self.encoder_updates:
            #     enc_loss = self.policy.train_acthead(b_eobs, b_enobs, b_acts)
           
            if itr % self.log_rate == 0:
                self.logger.log('density', ['loss'], [rnd_loss])
                self.logger.log('policy', ['actor_loss', 'critic_loss'], [actor_loss, critic_loss])
                # if itr < self.encoder_updates:
                #     self.logger.log('encoder', ['loss'], [enc_loss])
        
        # if encoder becomes in accurate then fine tune on new samples
        # last_enc_loss = self.encoder.get_loss(b_eobs, b_enobs, b_acts)
        # if (np.mean(last_enc_loss) > encoder_loss_thresh):
        #     self.train_enc_next_itr = True
        #     print('Updating Encoder....')
