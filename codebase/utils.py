import tensorflow as tf
import numpy as np 
import copy
import pickle
from pathlib import Path

def Network(input_tensor, output_size, scope, fsize, conv_depth, n_hidden_dense=0, activation=tf.tanh, output_activation=None):
        with tf.variable_scope(scope):
            x = input_tensor
            # Convolutions
            for _ in range(conv_depth):
                x = tf.layers.conv2d(x, fsize, (3, 3), activation='relu')
                x = tf.layers.conv2d(x, fsize, (3, 3), strides=(2, 2))
            
            # Dense Layers
            x = tf.layers.flatten(x)
            for _ in range(n_hidden_dense):
                x = tf.layers.dense(x, fsize, activation=activation)
            y = tf.layers.dense(x, output_size, activation=output_activation)
        return y

class ReplayBuffer(object):
    def __init__(self, max_size=10000):
        self.obs = []
        self.acts = []
        self.rewards = []
        self.nxt_obs = []
        self.dones = []
        self.logprobs = []
        self.max_size = max_size
    
    def record(self, obs, act, rew, nxt_ob, done):
        self.obs.append(obs)
        self.acts.append(act)
        self.rewards.append(rew)
        self.nxt_obs.append(nxt_ob)
        self.dones.append(done)
        
    def get_actions(self):
        return np.asarray(self.obs), np.asarray(self.acts)
    
    def set_logprobs(self, logprobs):
        self.logprobs += list(logprobs)
        assert len(self.logprobs) == len(self.obs), 'logprobs MUST == self.obs'
        
    def merge(self, obs, acts, rews, nxt_obs, dones, logprobs):
        self.obs += obs
        self.acts += acts
        self.rewards += rews
        self.nxt_obs += nxt_obs
        self.dones += dones
        self.logprobs += list(logprobs)
    
    def export(self):
        return self.obs, self.acts, self.rewards, self.nxt_obs, self.dones, self.logprobs
    
    def get_samples(self, indices):
        return (
            np.asarray(self.obs)[indices],
            np.asarray(self.acts)[indices],
            np.asarray(self.rewards)[indices],
            np.asarray(self.nxt_obs)[indices],
            np.asarray(self.dones)[indices],
            np.asarray(self.logprobs)[indices]
        )

    def get_all(self, batch_size):
        batched_dsets = []
        # batch up data
        for dset in [np.asarray(self.obs), np.asarray(self.acts), np.asarray(self.rewards), np.asarray(self.nxt_obs), np.asarray(self.dones), np.asarray(self.logprobs)]:
            bdset = []
            for i in range(0, len(dset), batch_size):
                 bdset.append(np.array(dset[i:i+batch_size]))
            batched_dsets.append(np.asarray(bdset))
        return tuple(batched_dsets)
    
    def update_size(self):
        diff = self.max_size - len(self)
        if diff < 0:
            # FIFO
            self.obs = self.obs[-diff:]
            self.acts = self.acts[-diff:]
            self.rewards = self.rewards[-diff:]
            self.nxt_obs = self.acts[-diff:]
            self.dones = self.dones[-diff:]
            self.logprobs = self.logprobs[-diff:]
    
    def flush(self):
        self.obs = []
        self.acts = []
        self.rewards = []
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
        
    def get_actions(self):
        return self.temp_replay.get_actions()
    
    def get_obs(self):
        return np.asarray(self.master_replay.obs)
    
    def set_logprobs(self, logprobs):
        self.temp_replay.logprobs = logprobs

    def merge_temp(self):
        tempdata = self.temp_replay.export()
        self.master_replay.merge(*tempdata)
        self.master_replay.update_size()
    
    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.master_replay), batch_size)
        return self.master_replay.get_samples(indices)

    def get_all_recent(self, batch_size):
        return self.temp_replay.get_all(batch_size)
    
    ## Density Sampling Start ##
    # credit to hw5
    def get_density_batch(self, states, batch_size):
        if len(self.master_replay) >= 2*len(states):
            positives, negatives = self.sample_idxs_replay(states, batch_size)
        else:
            positives, negatives = self.sample_idxs(states, batch_size)
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)
        return positives, negatives, labels
    
    def sample_idxs(self, states, batch_size):
        states = copy.deepcopy(states)
        data_size = len(states)
        pos_idxs = np.random.randint(data_size, size=batch_size)
        continue_sampling = True
        while continue_sampling:
            neg_idxs = np.random.randint(data_size, size=batch_size)
            if np.all(pos_idxs != neg_idxs):
                continue_sampling = False
        positives = np.concatenate([states[pos_idxs], states[pos_idxs]], axis=0)
        negatives = np.concatenate([states[pos_idxs], states[neg_idxs]], axis=0)
        return positives, negatives

    def sample_idxs_replay(self, states, batch_size):
        states = np.asarray(copy.deepcopy(states))
        data_size = len(states)
        pos_idxs = np.random.randint(data_size, size=batch_size)
        neg_idxs = np.random.randint(data_size, len(self.master_replay), size=batch_size)
        
        buffer_states = self.get_obs()
        positives = np.concatenate([states[pos_idxs], states[pos_idxs]], axis=0)
        negatives = np.concatenate([states[pos_idxs], buffer_states[neg_idxs]], axis=0)
        return positives, negatives
    ## Density Sampling End ##        
    
    def get_temp_reward_info(self):
        rewards = np.asarray(self.temp_replay.rewards)
        return np.sum(rewards), np.std(rewards)
        
    def flush_temp(self):
        self.temp_replay.flush()
    
    def flush(self):
        self.temp_replay.flush()
        self.master_replay.flush()

class Logger(object):
    def __init__(self):
        self.tag = None
        self.totalr = []
        self.std_reward = []
        
        self.tags = []
        self.results = []
        
        self.fset = Path('iter-frames')
        self.fset.mkdir(exist_ok=True)
        self.n_frames_stored = 0
        
    def set_tag(self, tag):
        self.tag = tag
        
    def log(self, totalr, std):
        self.totalr.append(totalr)
        self.std_reward.append(std)
        
    def log_frames(self, frames):
        fname = '{}-{}'.format(self.tag, self.n_frames_stored)
        with open(self.fset/fname, 'wb') as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        self.n_frames_stored += 1
        
    def package_results(self):
        # store
        self.tags.append(self.tag)
        self.results.append([self.totalr, self.std_reward])
    
    def flush(self):
        self.totalr = []
        self.std_reward = []
