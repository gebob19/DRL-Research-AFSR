import tensorflow as tf
import numpy as np 
from utils import Network

class DynamicsModel(object):
    def __init__(self, graph_args, rollout_args):
        self.enc_dim = graph_args['enc_dim']
        self.act_dim = graph_args['act_dim']
        self.act_low = graph_args['action_low']
        self.act_high = graph_args['action_high']
        
        self.learning_rate = graph_args['learning_rate']
        # network params
        self.hid_size = graph_args['hid_size']
        self.n_hidden = graph_args['n_hidden']
        # rollout args
        self.horizon = rollout_args['horizon']
        self.num_rollouts = rollout_args['num_rollouts']
        
        self.state, self.action, self.n_state = self.setup_placeholders()
        self.n_state_pred = self.dynamics_func(self.state, self.action, False)
        
        # Calculate Loss
        delta = self.state - self.n_state
        delta_pred = self.state - self.n_state_pred
        self.loss = tf.losses.mean_squared_error(delta, delta_pred)
        self.update_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def set_session(self, sess):
        self.sess = sess

    def setup_placeholders(self):
        state = tf.placeholder(shape=(None, self.enc_dim), name="state", dtype=tf.float32)
        action = tf.placeholder(shape=(None, self.act_dim), name='action', dtype=tf.float32)
        n_state = tf.placeholder(shape=(None, self.enc_dim), name="next_state", dtype=tf.float32)
        return state, action, n_state

    def dynamics_func(self, state, action, reuse):
        # add state, action normalization?
        sa = tf.concat([state, action], axis=1)
        delta_pred = Network(sa, self.enc_dim, 'dynamics', self.hid_size, conv_depth=0, n_hidden_dense=self.n_hidden, reuse=reuse)
        n_state_pred = state + delta_pred
        return n_state_pred
    
    def update(self, state, action, n_state):
        loss, _ = self.sess.run([self.loss, self.update_step], feed_dict={
            self.state: state,
            self.action: action,
            self.n_state: n_state
        })
        return loss


    def get_best_actions(self, state, profit_fcn, num_actions):
        """ Given encoded state will return `num_rollouts` rollouts where each rollout is of size `horizon`.
        Encoded state.
        => rollouts 
        => apply profit_fcn to rollouts 
        => choose rollout with highest profit 
        => return (first k-actions of best rollout, profit of rollout)
        """
        ran_sample = tf.random_uniform((self.horizon, self.num_rollouts, self.act_dim), 
                                       minval=self.act_low, 
                                       maxval=self.act_high, 
                                       dtype=tf.int32)
        
        rollout_profits = np.zeros((self.num_rollouts,), dtype=np.float32)
        
        # init state batch to starting state
        state_batch = tf.ones((self.num_rollouts, 1), dtype=tf.float32) * state
        for index in range(self.horizon):
            act_batch = tf.cast(ran_sample[index], tf.float32)
            next_state_batch = self.dynamics_func(state_batch, act_batch, True)
            state_batch = next_state_batch
            
            # check profit of current states for all rollouts 
            profit_batch = profit_fcn(state_batch, evaluate=True, encoded=True)
            rollout_profits += profit_batch
            
        max_profit_rollout = np.argmax(rollout_profits, axis=0)
        ran_sample = tf.transpose(ran_sample, perm=(1, 0, 2))
        best_actions = ran_sample[max_profit_rollout][:num_actions]
        return best_actions, rollout_profits[max_profit_rollout]