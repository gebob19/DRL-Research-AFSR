import tensorflow as tf
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from keras.applications.resnet50 import ResNet50
from keras import backend as K
K.set_learning_phase(1) 

from utils import Network

class Encoder(object):
    def __init__(self, graph_args):
        self.obs_dim = graph_args['obs_dim']
        self.act_dim = graph_args['act_dim']
        self.n_layers_frozen = graph_args['n_layers_frozen']
        self.act_layer_extract = graph_args['act_layer_extract']
        self.learning_rate = graph_args['learning_rate']

        self.fsize = graph_args['fsize']
        self.conv_depth = graph_args['conv_depth']
        self.n_strides = graph_args['n_strides']

        assert self.n_layers_frozen < self.act_layer_extract, 'Cannot freeze before action extraction'

        # action neural network params
        self.setup_action_classes()
        actnn_layers = graph_args['actnn_layers']
        actnn_units = graph_args['actnn_units']
        # encoding pass
        self.obs_ph = tf.placeholder(shape=((None,) + self.obs_dim), dtype=tf.float32) 
        self.obs_encoded = Network(self.obs_ph, None, 'encoder', self.fsize, self.conv_depth, activation=tf.nn.leaky_relu, n_strides=self.n_strides)

        # start of action neural net
        obs_enc_shape = self.obs_encoded.get_shape().as_list()
        # act_i-1, obs_i-1, obs_i placeholders
        self.prev_act_ph = tf.placeholder(shape=(None,), dtype=tf.int32) 
        self.actnn_prev_obs_ph = tf.placeholder(shape=obs_enc_shape, dtype=tf.float32)
        self.actnn_obs_ph = tf.placeholder(shape=obs_enc_shape, dtype=tf.float32)
        # concat & network pass
        multi_obs_enc = tf.concat([self.actnn_prev_obs_ph, self.actnn_obs_ph], axis=-1)
        self.actnn_pred = dense_pass(multi_obs_enc, self.act_dim, actnn_layers, actnn_units)
        
        action_enc = tf.one_hot(self.prev_act_ph, depth=self.act_dim)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actnn_pred, labels=action_enc)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def setup_action_classes(self):
        self.classes =  {
            '4': 0,  # look left action group = 0
            '7': 0,
            '9': 0,
            '12': 0,
            '15': 0,
            '17': 0,
            '3': 1,   # look right action group = 1
            '6': 1,
            '8': 1,
            '11': 1,
            '14': 1,
            '16': 1,
            '1': 2,    #  jump up action group = 2
            '10': 2,
            '2': 3,     # up / down action group = 3
            '5': 3,
            '13': 3,
            '0': 4,     #  do nothing action  group = 4
        }

    def set_sess(self, sess):
        self.sess = sess
    
    def get_encoding(self, obs_n):
        return self.sess.run(self.obs_encoded, feed_dict={
            self.obs_ph: obs_n
        })
    
    def train(self, obs_n, n_obs_n, act_n):
        # encode into action classes
        enc_act_n = self.encode_actions(act_n) 
        # update
        loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
            self.prev_act_ph: act_n,
            self.actnn_prev_obs_ph: obs_n,
            self.actnn_obs_ph: n_obs_n,
        })
        return loss

    def get_loss(self, obs_n, n_obs_n, act_n):
        # encode into action classes
        enc_act_n = self.encode_actions(act_n) 
        return self.sess.run(self.loss, feed_dict={
            self.prev_act_ph: act_n,
            self.actnn_prev_obs_ph: obs_n,
            self.actnn_obs_ph: n_obs_n,
        })
    
    def to_class(self, act):
        return self.classes[str(act)]

    def encode_actions(self, act_n):
        return np.array(list(map(self.to_class, act_n)))
    
    # def multi_t_resize(self, obs_n):
    #     with ThreadPoolExecutor(8) as e: pre_resized_obs = [e.submit(resize, obs) for obs in obs_n]
    #     resized_obs = np.asarray([obs.result() for obs in as_completed(pre_resized_obs)])
    #     return resized_obs

# def resize(obs):
#     return cv2.resize(obs, (224, 224))

class RND(object):
    def __init__(self, in_shape, graph_args):
        target_args, pred_args = graph_args['target_args'], graph_args['pred_args']
        self.learning_rate = graph_args['learning_rate']
        self.out_size = graph_args['out_size']
        self.proportion_to_update = graph_args['proportion_to_update']
        self.bonus_mean = graph_args['bonus_mean']
        self.bonus_var = graph_args['bonus_var']
        self.bonus_multi = graph_args['bonus_multiplier']

        self.enc_obs = tf.placeholder(shape=in_shape, dtype=tf.float32)

        # f(o*) and f*(o*)
        self.target_output = self.set_network(self.enc_obs, target_args, 'rnd_targetN')
        self.pred_output = self.set_network(self.enc_obs, pred_args, 'rnd_predictorN')
        
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(self.target_output) - self.pred_output), axis=-1)

        self.aux_loss = tf.reduce_mean(tf.square(tf.stop_gradient(self.target_output) - self.pred_output), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_to_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.aux_loss)

    def set_network(self, ph, args, scope):
        fsize = args['fsize']
        conv_depth = args['conv_depth']
        n_layers = args['n_layers']
        kernel_init = args['kernel_init']
        
        network_output = Network(ph, self.out_size, scope, fsize, conv_depth, n_layers, n_strides=2, kernel_init=kernel_init)
        return network_output

    def set_sess(self, sess):
        self.sess = sess

    def get_rewards(self, enc_obs_n):
        return self.sess.run(self.int_rew, feed_dict={
            self.enc_obs: enc_obs_n,
        })

    def modify_rewards(self, enc_obs_n, rewards):
        extr_rewards = self.get_rewards(enc_obs_n)
        # norm_extr_rew = (extr_rewards - np.mean(extr_rewards)) / (np.var(extr_rewards) + 1e-6)
        # norm_extr_rew = self.bonus_mean + norm_extr_rew * self.bonus_var
        return rewards + self.bonus_multi * extr_rewards
    
    def train(self, enc_obs_n):
        loss, _ = self.sess.run([self.aux_loss, self.update_op], feed_dict={
            self.enc_obs: enc_obs_n,
        })
        return loss  

def dense_pass(x, out_size, num_layers, units, output_activation=None):
    x = tf.layers.flatten(x)
    for _ in range(num_layers):
        x = tf.layers.dense(x, units, activation=tf.tanh)
    return tf.layers.dense(x, out_size, activation=output_activation)
