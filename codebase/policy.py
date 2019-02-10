import tensorflow as tf
import numpy as np 
from utils import Network

class PPO(object):
    def __init__(self, graph_args, adv_args, in_shape):
        self.act_dim = graph_args['act_dim']
        # clip_range = graph_args['clip_range']
        # conv operations params
        n_hidden = graph_args['n_hidden']
        hid_size = graph_args['hid_size']

        conv_depth = graph_args['conv_depth']
        
        self.learning_rate = graph_args['learning_rate']
        self.num_target_updates = graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = graph_args['num_grad_steps_per_target_update']
        
        self.gamma = adv_args['gamma']

        self.act, self.adv = self.define_placeholders()
        self.obs = tf.placeholder(shape=in_shape, dtype=tf.float32)
        self.n_obs = tf.placeholder(shape=in_shape, dtype=tf.float32)
        
        # policy / actor evaluation with encoded state
        self.half_policy_distrib = Network(self.obs, None, 'policy_start', \
            hid_size, conv_depth)
        self.half_policy_distrib_2 = Network(self.n_obs, None, 'policy_start', \
            hid_size, conv_depth, reuse=True)

        self.policy_distrib = Network(self.half_policy_distrib, self.act_dim,  \
            'policy_out', hid_size, n_hidden_dense=n_hidden)

        self.greedy_action = tf.argmax(self.policy_distrib, axis=1)

        # U = tf.random_uniform(tf.shape(self.policy_distrib), minval = 0, maxval = 1)
        # self.sample_action = tf.argmax(self.policy_distrib - tf.log(-tf.log(U)), axis=1)
        self.n_act_sample = 1
        self.sample_action = tf.random.multinomial(tf.nn.softmax(self.policy_distrib), self.n_act_sample)
        
        # policy update
        action_enc = tf.one_hot(self.act, depth=self.act_dim)
        self.logprob = -1 * tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy_distrib, labels=action_enc)
        self.actor_loss = -tf.reduce_mean(self.logprob * self.adv - 2e-1 * self.logprob)
        actor_optim =  tf.train.AdamOptimizer(self.learning_rate)
        self.grads = actor_optim.compute_gradients(self.actor_loss)
        for grad in self.grads:
            tf.summary.histogram("{}-grad".format(grad[1].name), grad)

        self.merged = tf.summary.merge_all()

        self.actor_update_op = actor_optim.minimize(self.actor_loss)
        
        # critic definition with encoded state
        self.v_target = tf.placeholder(shape=(None,), name='v_target', dtype=tf.float32)
        self.v_pred = tf.squeeze(Network(self.obs, 1, 'critic', hid_size, conv_depth=conv_depth, n_hidden_dense=n_hidden))
        self.critic_loss = tf.losses.mean_squared_error(self.v_target, self.v_pred)
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

        # action neural network def
        # act_i-1, obs_i-1, obs_i placeholders
        actnn_layers = graph_args['actnn_layers']
        actnn_units = graph_args['actnn_units']
        self.actnn_learning_rate = graph_args['actnn_learning_rate']

        self.prev_act_ph = tf.placeholder(shape=(None,), dtype=tf.int32) 
        self.actnn_prev_obs_ph = self.half_policy_distrib
        self.actnn_obs_ph = self.half_policy_distrib_2
        # concat & network pass
        multi_obs_enc = tf.concat([self.actnn_prev_obs_ph, self.actnn_obs_ph], axis=-1)
        self.actnn_pred = dense_pass(multi_obs_enc, self.act_dim, actnn_layers, actnn_units)
        
        action_enc = tf.one_hot(self.prev_act_ph, depth=self.act_dim)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actnn_pred, labels=action_enc)
        self.train_step = tf.train.AdamOptimizer(self.actnn_learning_rate).minimize(self.loss)
        
    def set_session(self, sess):
        self.sess = sess
        
    def define_placeholders(self):
        act = tf.placeholder(shape=(None,), name='act', dtype=tf.int32)
        adv = tf.placeholder(shape=(None,), name='adv', dtype=tf.float32)
        return act, adv

    def sample(self, enc_obs):
        # obs must be shape (1, ob_dim)
        self.n_act_sample = len(enc_obs)
        return self.sess.run(self.sample_action, feed_dict={
            self.obs: enc_obs
        })[0][0]
        
    def get_best_action(self, enc_obs):
        # obs must be shape (1, ob_dim)
        return self.sess.run(self.greedy_action, feed_dict={
            self.obs: enc_obs
        })[0]

    def get_act_distrib(self, obs):
        return self.sess.run(self.policy_distrib, feed_dict={
            self.obs: [obs]
        })[0]
    
    def estimate_adv(self, obs, rew, nxt_obs, dones):
        v_obs = self.sess.run(self.v_pred, feed_dict={self.obs: obs})
        v_nxt_obs = self.sess.run(self.v_pred, feed_dict={self.obs: nxt_obs})
        y_obs = rew + (1 - dones) * self.gamma * v_nxt_obs
        adv = y_obs - v_obs
        
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        return adv
    
    def train_actor(self, obs, act, adv):
        loss, _, merged = self.sess.run([self.actor_loss, self.actor_update_op, self.merged], feed_dict={
            self.obs: obs,
            self.act: act,
            self.adv: adv
        })
        return loss, merged
        
    def train_critic(self, obs, nxt_obs, rew, dones):
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                v_pred = self.sess.run(self.v_pred, feed_dict={self.obs: nxt_obs})
                y = rew + self.gamma * v_pred * (1 - dones)
            _, loss = self.sess.run([self.critic_update_op, self.critic_loss],
                                    feed_dict={self.obs: obs, self.v_target: y})
        return loss


    def train_acthead(self, obs_n, n_obs_n, act_n):
        loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
            self.prev_act_ph: act_n,
            self.obs: obs_n,
            self.n_obs: n_obs_n
        })
        return loss

def dense_pass(x, out_size, num_layers, units, output_activation=None):
    x = tf.layers.flatten(x)
    for _ in range(num_layers):
        x = tf.layers.dense(x, units, activation=tf.tanh)
    return tf.layers.dense(x, out_size, activation=output_activation)