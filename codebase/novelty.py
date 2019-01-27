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
        self.act_dim = graph_args['act_dim']
        self.n_layers_frozen = graph_args['n_layers_frozen']
        self.act_layer_extract = graph_args['act_layer_extract']
        self.learning_rate = graph_args['learning_rate']

        assert self.n_layers_frozen < self.act_layer_extract, 'Cannot freeze before action extraction'

        # action neural network params
        actnn_layers = graph_args['actnn_layers']
        actnn_units = graph_args['actnn_units']

        self.obs_ph = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32) # obs_i
        self.prev_act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)  # act_i-1

        nn = ResNet50(include_top=False, weights='imagenet', input_tensor=self.obs_ph)
        self.freeze(nn)
                
        self.obs_encoded = nn.layers[self.act_layer_extract].output
        self.actnn_pred = dense_pass(self.obs_encoded, self.act_dim, actnn_layers, actnn_units)
        
        action_enc = tf.one_hot(self.prev_act_ph, depth=self.act_dim)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actnn_pred, labels=action_enc)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def freeze(self, x):
        for layer in x.layers[:self.n_layers_frozen]:
            layer.trainable = False
    
    def set_sess(self, sess):
        self.sess = sess
    
    def get_encoding(self, obs_n):
        resized_obs = self.multi_t_resize(obs_n)
        return self.sess.run(self.obs_encoded, feed_dict={
            self.obs_ph: resized_obs 
        })
    
    def train(self, obs_n, prev_act_n):
        resized_obs = self.multi_t_resize(obs_n)
        loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
            self.obs_ph: resized_obs,
            self.prev_act_ph: prev_act_n
        })
        return loss
    
    def multi_t_resize(self, obs_n):
        with ThreadPoolExecutor(8) as e: pre_resized_obs = [e.submit(resize, obs) for obs in obs_n]
        resized_obs = np.asarray([obs.result() for obs in as_completed(pre_resized_obs)])
        return resized_obs

def resize(obs):
    return cv2.resize(obs, (224, 224))

class RND(object):
    def __init__(self, in_shape, graph_args):
        target_args, pred_args = graph_args['target_args'], graph_args['pred_args']
        self.learning_rate = graph_args['learning_rate']
        self.bonus_multi = graph_args['bonus_multiplier']
        self.out_size = graph_args['out_size']

        self.enc_obs = tf.placeholder(shape=in_shape, dtype=tf.float32)

        # f(o*) and f*(o*)
        self.target_output = self.set_network(self.enc_obs, target_args, 'rnd_targetN')
        self.pred_output = self.set_network(self.enc_obs, pred_args, 'rnd_predictorN')
        # print(self.target_output, self.pred_output)
        self.loss = tf.losses.mean_squared_error(self.target_output, self.pred_output)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def set_network(self, ph, args, scope):
        fsize = args['fsize']
        conv_depth = args['conv_depth']
        n_layers = args['n_layers']
        kernel_init = args['kernel_init']
        
        # TODO make the weights more random => loss is starting at 0.08 ffs
        network_output = Network(ph, self.out_size, scope, fsize, conv_depth, n_layers, n_strides=2, kernel_init=kernel_init)
        return network_output

    def set_sess(self, sess):
        self.sess = sess

    def get_rewards(self, enc_obs_n):
        return self.sess.run(self.loss, feed_dict={
            self.enc_obs: enc_obs_n,
        })

    def modify_rewards(self, enc_obs_n, rewards):
        extr_rewards = self.get_rewards(enc_obs_n)
        return rewards + self.bonus_multi * extr_rewards
    
    def train(self, enc_obs_n):
        loss, _ = self.sess.run([self.loss, self.update_op], feed_dict={
            self.enc_obs: enc_obs_n,
        })
        return loss  

def dense_pass(x, out_size, num_layers, units, output_activation=None):
    x = tf.layers.flatten(x)
    for _ in range(num_layers):
        x = tf.layers.dense(x, units, activation=tf.tanh)
    return tf.layers.dense(x, out_size, activation=output_activation)