from keras.applications.resnet50 import ResNet50
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from keras import backend as K
K.set_learning_phase(1) 

class Encoder(object):
    def __init__(self, extract_layer, weights, freeze, scope):
        self.extract_layer = extract_layer
        # build graph
        self.x = tf.keras.backend.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
        x = ResNet50(include_top=False, weights=weights, input_tensor=self.x)
        if freeze: self.freeze(x)
                
        x = x.layers[self.extract_layer].output
        self.encoded_tensor = tf.layers.flatten(x)

    def freeze(self, x):
        for layer in x.layers[:self.extract_layer+1]: 
            layer.trainable = False 
    
    def set_sess(self, sess):
        self.sess = sess
    
    def get_encoding(self, obs_n):
        resized_obs = self.multi_t_resize(obs_n)
        return self.sess.run(self.encoded_tensor, feed_dict={
            self.x: resized_obs
        })
    
    def multi_t_resize(self, obs_n):
        with ThreadPoolExecutor(8) as e: pre_resized_obs = [e.submit(resize, obs) for obs in obs_n]
        resized_obs = np.asarray([obs.result() for obs in as_completed(pre_resized_obs)])
        return resized_obs

def resize(obs):
    return cv2.resize(obs, (224, 224))

class RND(object):
    def __init__(self, target_network, rn_layer, graph_args):
        self.learning_rate = graph_args['learning_rate']
        out_size = graph_args['out_size']
        n_layers = graph_args['n_layers']
        n_hidden = graph_args['n_hidden']
        self.bonus_multi = graph_args['bonus_multiplier']
        
        assert isinstance(target_network, Encoder), 'Target Network must be Encoder'

        self.target_network = target_network
        self.random_network = Encoder(rn_layer, None, False, 'RND')

        assert n_layers % 2 == 0, 'rnd n_layers must be divisible by 2'

        self.target_output = dense_pass(self.target_network.encoded_tensor, out_size, n_layers//2, n_hidden)
        self.rn_pred = dense_pass(self.random_network.encoded_tensor, out_size, n_layers, n_hidden)

        self.loss = tf.losses.mean_squared_error(self.target_output, self.rn_pred)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def set_sess(self, sess):
        self.sess = sess

    def get_rewards(self, obs):
        return self.sess.run(self.loss, feed_dict={
            self.target_network.x: obs,
            self.random_network.x: obs
        })

    def modify_rewards(self, obs, rewards):
        extr_rewards = self.get_rewards(obs)
        return rewards + self.bonus_multi * extr_rewards
    
    def train(self, obs):
        loss, _ = self.sess.run([self.loss, self.update_op], feed_dict={
            self.target_network.x: obs,
            self.random_network.x: obs
        })
        return loss  

def dense_pass(x, out_size, num_layers, n_hidd):
    for _ in range(num_layers):
        x = tf.layers.dense(x, n_hidd, activation=None)
    return tf.layers.dense(x, out_size, activation=None)