import tensorflow as tf
import numpy as np 
from utils import Network

class Policy(object):
    def __init__(self, graph_args, adv_args):
        self.ob_dim = graph_args['ob_dim']
        self.act_dim = graph_args['act_dim']
        clip_range = graph_args['clip_range']
        # conv operations params
        conv_depth = graph_args['conv_depth']
        filter_size = graph_args['filter_size']
        
        self.learning_rate = graph_args['learning_rate']
        self.num_target_updates = graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = graph_args['num_grad_steps_per_target_update']
        
        self.gamma = adv_args['gamma']
        
        self.obs, self.act, self.adv, self.old_logprob = self.define_placeholders()
        
        # policy / actor evaluation
        self.policy_distrib = Network(self.obs, self.act_dim, 'policy', filter_size, conv_depth)
        self.greedy_action = tf.argmax(self.policy_distrib, axis=1)
        self.logprob = self.get_logprob(self.policy_distrib, self.act)
        
        # importance sampling
        ratio = tf.exp(self.logprob - self.old_logprob)
        clipped_ratio = tf.clip_by_value(ratio, 1.0-clip_range, 1.0+clip_range)        
        # include increase entropy term with alpha=0.2
        batch_loss = tf.minimum(ratio*self.adv, clipped_ratio * self.adv) - 0.2 * self.logprob
        self.actor_loss = -tf.reduce_mean(batch_loss)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)
        
        # critic definition
        self.v_pred = tf.squeeze(Network(self.obs, 1, 'critic', filter_size, conv_depth, n_hidden_dense=2))
        self.v_target = tf.placeholder(shape=(None,), name='v_target', dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.v_target, self.v_pred)
        # minimize with respect to correct variables HERE
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)
        
    def set_session(self, sess):
        self.sess = sess
        
    def define_placeholders(self):
        obs = tf.placeholder(shape=((None,) + (self.ob_dim)), name='obs', dtype=tf.float32)
        act = tf.placeholder(shape=(None,), name='act', dtype=tf.int32)
        adv = tf.placeholder(shape=(None,), name='adv', dtype=tf.float32)
        logprob = tf.placeholder(shape=(None,), name='logprob', dtype=tf.float32)
        return obs, act, adv, logprob
    
    def get_logprob(self, policy_distribution, actions):
        action_enc = tf.one_hot(actions, depth=self.act_dim)
        logprob = -1 * tf.nn.softmax_cross_entropy_with_logits_v2(logits=policy_distribution, labels=action_enc)
        return logprob
        
    def get_best_action(self, observation):
        act = self.sess.run(self.greedy_action, feed_dict={
            self.obs: [observation]
        })[0]
        return act
    
    def estimate_adv(self, obs, rew, nxt_obs, dones):
        ## Markov Implementation??
        # V(s) & V(s')
        v_obs = self.sess.run(self.v_pred, feed_dict={self.obs: obs})
        v_nxt_obs = self.sess.run(self.v_pred, feed_dict={self.obs: nxt_obs})
        # y = r + gamma * V(s')
        y_obs = rew + (1 - dones) * self.gamma * v_nxt_obs
        # Adv(s) = y - V(s)
        adv = y_obs - v_obs
        # Normalize advantages
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        return adv
    
    def train_actor(self, obs, act, logprob, adv):
        loss, _ = self.sess.run([self.actor_loss, self.actor_update_op], feed_dict={
            self.obs: obs,
            self.act: act,
            self.adv: adv,
            self.old_logprob: logprob
        })
        return loss
        
    def train_critic(self, obs, nxt_obs, rew, dones):
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                v_pred = self.sess.run(self.v_pred, feed_dict={self.obs: nxt_obs})
                y = rew + self.gamma * v_pred * (1 - dones)
            _, loss = self.sess.run([self.critic_update_op, self.critic_loss],
                                    feed_dict={self.obs: obs, self.v_target: y})
        return loss