import tensorflow as tf
import numpy as np 
from utils import Network

class PPO(object):
    def __init__(self, graph_args, adv_args):
        self.ob_dim = graph_args['ob_dim']
        self.act_dim = graph_args['act_dim']
        clip_range = graph_args['clip_range']
        # conv operations params
        n_hidden = graph_args['n_hidden']
        hid_size = graph_args['hid_size']
        conv_depth = graph_args['conv_depth']
        
        self.learning_rate = graph_args['learning_rate']
        self.num_target_updates = graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = graph_args['num_grad_steps_per_target_update']
        
        self.gamma = adv_args['gamma']
        
        self.act, self.adv, self.old_logprob = self.define_placeholders()
        self.obs = tf.placeholder(shape=((None,) + self.ob_dim), dtype=tf.float32)
        
        # policy / actor evaluation with encoded state
        self.policy_distrib = Network(self.obs, self.act_dim, 'policy', hid_size, conv_depth, n_hidden)
        self.greedy_action = tf.argmax(self.policy_distrib, axis=1)

        action_enc = tf.one_hot(self.act, depth=self.act_dim)
        self.logprob = -1 * tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy_distrib, labels=action_enc)
        
        # importance sampling
        ratio = tf.exp(self.logprob - self.old_logprob)
        clipped_ratio = tf.clip_by_value(ratio, 1.0-clip_range, 1.0+clip_range)        
        # include increase entropy term with alpha=0.2
        batch_loss = tf.minimum(ratio*self.adv, clipped_ratio * self.adv) #- 0.2 * self.logprob
        self.actor_loss = -tf.reduce_mean(batch_loss)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)
        
        # critic definition with encoded state
        self.v_pred = tf.squeeze(Network(self.obs, 1, 'critic', hid_size, n_hidden_dense=n_hidden))
        self.v_target = tf.placeholder(shape=(None,), name='v_target', dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.v_target, self.v_pred)
        # minimize with respect to correct variables HERE
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)
        
    def set_session(self, sess):
        self.sess = sess
        
    def define_placeholders(self):
        act = tf.placeholder(shape=(None,), name='act', dtype=tf.int32)
        adv = tf.placeholder(shape=(None,), name='adv', dtype=tf.float32)
        logprob = tf.placeholder(shape=(None,), name='logprob', dtype=tf.float32)
        return act, adv, logprob
        
    def get_best_action(self, enc_obs):
        # obs must be shape (1, ob_dim)
        return self.sess.run(self.greedy_action, feed_dict={
            self.obs: enc_obs
        })[0]

    def get_act_distrib(self, obs):
        return self.sess.run(self.policy_distrib, feed_dict={
            self.obs: [obs]
        })[0]

    def get_logprob(self, n_obs, n_act):
        return self.sess.run(self.logprob, feed_dict={
            self.obs: n_obs,
            self.act: n_act
        })
    
    def estimate_adv(self, obs, rew, nxt_obs, dones):
        v_obs = self.sess.run(self.v_pred, feed_dict={self.obs: obs})
        v_nxt_obs = self.sess.run(self.v_pred, feed_dict={self.obs: nxt_obs})
        y_obs = rew + (1 - dones) * self.gamma * v_nxt_obs
        adv = y_obs - v_obs
        
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