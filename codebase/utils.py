import tensorflow as tf
import numpy as np 
import gym
from gym import spaces
import cv2

import copy
import datetime
import pickle
from random import shuffle
from pathlib import Path

def Network(input_tensor, output_size, scope, fsize, conv_depth=0, n_hidden_dense=0, 
            activation=tf.tanh, output_activation=None, reuse=False, n_strides=None, kernel_init=None):
        with tf.variable_scope(scope, reuse=reuse):
            x = input_tensor
            if n_strides == None: n_strides = conv_depth
            strides_count = 0
            # Convolutions
            for _ in range(conv_depth):
                x = tf.layers.conv2d(x, fsize, (3, 3), activation='relu', kernel_initializer=kernel_init)
                if strides_count < n_strides:
                    x = tf.layers.conv2d(x, fsize, (3, 3), strides=(2, 2), kernel_initializer=kernel_init)
                    strides_count += 1
            # Dense Layers
            if output_size is not None:
                x = tf.layers.flatten(x)
                for _ in range(n_hidden_dense):
                    x = tf.layers.dense(x, fsize, activation=activation, kernel_initializer=kernel_init)
                y = tf.layers.dense(x, output_size, activation=output_activation, kernel_initializer=kernel_init)
            else:
                y = x
        return y

def make_env(env_id, width, height):
    env = gym.make(env_id)
    env = WarpFrame(env, width, height)
    env = StickyActionEnv(env)
    return env

## https://github.com/openai/random-network-distillation/blob/master/atari_wrappers.py
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

# Credits to: https://github.com/jcwleo/random-network-distillation-pytorch/blob/84dd0544de602c28f8f277d4ae627a1c4aa0af36/utils.py
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
class RunningMeanStd(object):
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = ((self.count * self.mean) + (batch_count * batch_mean)) / tot_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class Logger(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.size = 0
        self.logs = {
            'density': {
                'loss': []
            },
            'encoder': {
                'loss': []
            },
            'policy': {
                'actor_loss':[],
                'critic_loss':[],
            },
            'env': {
                'int_rewards': [],
                'ext_rewards': [],
                'frames': [],
                'norm_int_rew': []
            }
        }
        self.model_name = ''
        
    def log(self, tag, subtags, data):
        for subtag, d in zip(subtags, data):
            self.logs[tag][subtag].append(d)
        self.size += 1
        
        if self.size > self.max_size:
            self.export()
            self.flush()
            self.size = 0

    def export(self):
        fname = '{}-{}.pkl'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.model_name)
        with open(fname, 'wb') as f:
            pickle.dump(self.logs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def import_logs(self, fname):
        with open(fname, 'rb') as f:
            self.logs = pickle.load(f)
    
    def flush(self):
        self.logs = {
            'density': {
                'loss': []
            },
            'encoder': {
                'loss': []
            },
            'policy': {
                'actor_loss':[],
                'critic_loss':[],
            },
            'env': {
                'int_rewards': [],
                'ext_rewards': []
            }
        }



## NOT USED 
class ReplayBuffer(object):
    def __init__(self, max_size=10000):
        self.obs = []
        self.acts = []
        self.ext_rewards = []
        self.int_rewards = []
        self.nxt_obs = []
        self.dones = []
        self.logprobs = []
        self.max_size = max_size
    
    def record(self, obs, act, rew, nxt_ob, done):
        self.obs.append(obs)
        self.acts.append(act)
        self.ext_rewards.append(rew)
        self.nxt_obs.append(nxt_ob)
        self.dones.append(done)
        
    def get_logger_work(self):
        return np.array(self.obs), np.array(self.acts), np.array(self.ext_rewards)
    
    def set_logprobs(self, logprobs):
        self.logprobs += list(logprobs)
        assert len(self.logprobs) == len(self.obs), 'logprobs MUST == self.obs'

    def set_intrew(self, int_rew):
        self.int_rewards += list(int_rew)
        assert len(self.int_rewards) == len(self.ext_rewards), 'len int rew MUST == len ext rew'

    def merge(self, obs, acts, ext_rews, int_rews, nxt_obs, dones, logprobs):
        self.obs += obs
        self.acts += acts
        self.ext_rewards += ext_rews
        self.int_rewards += int_rews
        self.nxt_obs += nxt_obs
        self.dones += dones
        self.logprobs += list(logprobs)
    
    def export(self):
        return self.obs, self.acts, self.ext_rewards, self.int_rewards, self.nxt_obs, self.dones, self.logprobs
    
    def get_samples(self, indices):
        return (
            np.array(self.obs)[indices],
            np.array(self.acts)[indices],
            np.array(self.ext_rewards)[indices],
            np.array(self.int_rewards)[indices],
            np.array(self.nxt_obs)[indices],
            np.array(self.dones)[indices],
            np.array(self.logprobs)[indices]
        )

    def get_all(self, batch_size, shuffle=False, size=None):
        o, a, er, ir, n, d, l = np.array(self.obs), np.array(self.acts), np.array(self.ext_rewards), np.array(self.int_rewards), np.array(self.nxt_obs), np.array(self.dones), np.array(self.logprobs)
        if shuffle:
            indxs = np.arange(o.shape[0])
            np.random.shuffle(indxs)
            o, a, er, ir, n, d, l = o[indxs], a[indxs], er[indxs], ir[indxs], n[indxs], d[indxs], l[indxs]
        if size is not None:
            o, a, er, ir, n, d, l = o[-size:], a[-size:], er[-size:], ir[-size:], n[-size:], d[-size:], l[-size:]
        batched_dsets = []
        # batch up data
        for dset in [o, a, er, ir, n, d, l]:
            bdset = []
            for i in range(0, len(dset), batch_size):
                 bdset.append(np.array(dset[i:i+batch_size]))
            batched_dsets.append(np.array(bdset))
        return tuple(batched_dsets)
    
    def update_size(self):
        diff = self.max_size - len(self)
        if diff < 0:
            # FIFO
            self.obs = self.obs[-diff:]
            self.acts = self.acts[-diff:]
            self.ext_rewards = self.ext_rewards[-diff:]
            self.int_rewards = self.int_rewards[-diff:]
            self.nxt_obs = self.nxt_obs[-diff:]
            self.dones = self.dones[-diff:]
            self.logprobs = self.logprobs[-diff:]
    
    def flush(self):
        self.obs = []
        self.acts = []
        self.ext_rewards = []
        self.int_rewards = []
        self.nxt_obs = []
        self.dones = []
        self.logprobs = []
    
    def __len__(self):
        return len(self.obs)
        
class MasterBuffer(object):
    def __init__(self, max_size):
        self.master_replay = ReplayBuffer(max_size)
        self.temp_replay = ReplayBuffer()
    
    def record(self, *args):
        self.temp_replay.record(*args)
        
    def get_logger_work(self):
        return self.temp_replay.get_logger_work()
    
    def get_obs(self):
        return np.array(self.master_replay.obs)
    
    def set_other(self, logprobs, int_rews):
        self.temp_replay.logprobs = logprobs
        self.temp_replay.int_rewards = int_rews

    def merge_temp(self):
        tempdata = self.temp_replay.export()
        self.master_replay.merge(*tempdata)
        self.master_replay.update_size()

    def get_all(self, batch_size, master=False, shuffle=False, size=None):
        if not master:
            return self.temp_replay.get_all(batch_size, shuffle)
        else:
            return self.master_replay.get_all(batch_size, shuffle, size)    
        
    def flush_temp(self):
        self.temp_replay.flush()
    
    def flush(self):
        self.temp_replay.flush()
        self.master_replay.flush()

