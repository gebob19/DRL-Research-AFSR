import tensorflow as tf
import numpy as np

from utils import RunningMeanStd

class Agent(object):
    def __init__(self, env, policy, rnd, replay_buffer, logger, args):
        self.env = env
        # Models
        self.policy = policy
        self.rnd = rnd
        # Utils
        self.replay_buffer = replay_buffer
        self.logger = logger
        
        self.obs_running_mean = RunningMeanStd((84, 84, 1))
        self.rew_running_mean = RunningMeanStd(())

        self.last_enc_loss = None
        self.train_enc_next_itr = False

        # Args
        self.rnd_train_itr = args['rnd_train_itr']
        self.use_encoder = args['use_encoder']
        self.encoder_train_itr = args['encoder_train_itr']
        self.encoder_train_limit = args['encoder_train_limit']

        self.num_random_samples = args['num_random_samples']
        self.log_rate = args['log_rate']
        
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)
        self.rnd.set_sess(sess)

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
        n_lives, ignore = 6, 0
        obs_n, act_n, ext_rew_n, int_rew_n, n_obs_n, dones_n = [], [], [], [], [], []
        
        # policy rollout
        obs = self.env.reset()
        while not done and i < num_samples:
            if algorithm == 'algorithm' and ignore < 0:
                act = self.policy.sample([obs])
            else: # algorithm == 'random'
                act = self.env.action_space.sample()

            n_obs, rew, done, info = self.env.step(act)
            int_rew = self.rnd.get_rewards([obs])[0]
            
            # dont record when agent dies
            if info['ale.lives'] != n_lives: ignore = 18; n_lives -= 1
            if not ignore:
                i += 1
                self.rew_running_mean.update(np.array([int_rew]))

                obs_n.append(obs); ext_rew_n.append(rew); n_obs_n.append(n_obs)
                act_n.append(act); dones_n.append(done); int_rew_n.append(int_rew)
                if done:
                    obs = self.env.reset()
                    done = True
                    n_lives, ignore = 6, 0
            else: ignore -= 1

        # normalize
        int_rew_n = (int_rew_n - self.rew_running_mean.mean) / self.rew_running_mean.var
        ext_rew_n = np.clip(ext_rew_n, -1, 1)

        self.obs_running_mean.update(np.array(obs_n))

        self.logger.log('env', ['int_rewards', 'ext_rewards'], [int_rew_n, ext_rew_n])
        return obs_n, act_n, ext_rew_n, int_rew_n, n_obs_n, dones_n
        
    def get_data(self, batch_size, num_samples, itr):
        if itr < self.num_random_samples:
            return self.sample_env(batch_size, num_samples, shuffle=True, algorithm='random')
        return self.sample_env(batch_size, num_samples, shuffle=True)

    def init_obsmean(self):
        obs, done = self.env.reset(), False
        while not done:
            act = self.env.action_space.sample()
            obs, _, done, _  = self.env.step(act)
            self.obs_running_mean.update(obs)

    def init_encoder(self, batch_size, num_samples, loss_threshold):
        threshold_met, i = False, 0
        losses = []

        while not threshold_met and i < self.encoder_train_limit:
            raw_enc_obs, raw_act_n, raw_ext_rew_n, raw_int_rew, raw_enc_n_obs, raw_dones_n  = self.sample_env(batch_size, num_samples, shuffle=True, algorithm='random')
            for _ in range(self.encoder_train_itr):
                enc_obs, act_n, _, _, enc_n_obs, _ = self.batch(raw_enc_obs, raw_act_n, raw_ext_rew_n, raw_int_rew, raw_enc_n_obs, raw_dones_n, batch_size, shuffle=True)
                for b_eobs, b_acts, b_enobs in zip(enc_obs, act_n, enc_n_obs):

                    enc_loss = self.policy.train_acthead(b_eobs, b_enobs, b_acts)
                    losses.append(np.mean(enc_loss))
                    self.logger.log('encoder', ['loss'], [np.mean(enc_loss)])
                    i += 1

                if np.mean(losses) < loss_threshold: threshold_met = True
                losses = []

        if threshold_met: print('Encoder init threshold was met...')
        else: print('Encoder init threshold was NOT met...')
    
    def train(self, batch_size, num_samples, encoder_loss_thresh, itr, writer):
        raw_enc_obs, raw_act_n, raw_ext_rew_n, raw_int_rew, raw_enc_n_obs, raw_dones_n = self.get_data(batch_size, num_samples, itr)
        
        for _ in range(4):
            # reshuffle and batch 
            enc_obs, act_n, ext_rew_n, int_rew, enc_n_obs, dones_n = self.batch(raw_enc_obs, raw_act_n, raw_ext_rew_n, raw_int_rew, raw_enc_n_obs, raw_dones_n, batch_size, shuffle=True)
            for b_eobs, b_acts, b_erew, b_irew, b_enobs, b_dones in zip(enc_obs, act_n, ext_rew_n, int_rew, enc_n_obs, dones_n):
                
                # norm and clip for rnd
                rnd_obs = (b_eobs - self.obs_running_mean.mean) / self.obs_running_mean.var
                rnd_obs = np.clip(rnd_obs, -5, 5)
                rnd_loss = self.rnd.train(rnd_obs)
                
                total_r = b_erew + b_irew
                
                # 1 critic temp soln
                critic_loss = self.policy.train_critic(b_eobs, b_enobs, total_r, b_dones)
                adv = self.policy.estimate_adv(b_eobs, total_r, b_enobs, b_dones)
                actor_loss, summ = self.policy.train_actor(b_eobs, b_acts, adv)
                writer.add_summary(summ, itr)

                if self.use_encoder and self.train_enc_next_itr:
                    enc_loss = self.policy.train_acthead(b_eobs, b_enobs, b_acts)
                    self.logger.log('encoder', ['loss'], [enc_loss])
            
                if itr % self.log_rate == 0:
                    self.logger.log('density', ['loss'], [rnd_loss])
                    self.logger.log('policy', ['actor_loss', 'critic_loss'], [actor_loss, critic_loss])
            
        self.train_enc_next_itr = False
        # if encoder becomes in accurate then fine tune next train iteration
        if self.use_encoder:
            enc_loss = self.policy.actnn_loss(b_eobs, b_enobs, b_acts)
            if np.mean(enc_loss) > encoder_loss_thresh:
                self.train_enc_next_itr = True
                print('Updating Encoder....')
