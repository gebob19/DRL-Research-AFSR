import tensorflow as tf
from utils import Network

class RND(object):
    def __init__(self, in_shape, graph_args):
        target_args, pred_args = graph_args['target_args'], graph_args['pred_args']
        self.learning_rate = graph_args['learning_rate']
        self.out_size = graph_args['out_size']
        self.proportion_to_update = graph_args['proportion_to_update']

        self.enc_obs = tf.placeholder(shape=in_shape, dtype=tf.float32)

        # f(o*) and f*(o*)
        self.target_output = self.set_network(self.enc_obs, target_args, 'rnd_targetN')
        self.pred_output = self.set_network(self.enc_obs, pred_args, 'rnd_predictorN')
        
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(self.target_output) - self.pred_output), axis=-1)

        self.aux_loss = tf.reduce_mean(tf.square(tf.stop_gradient(self.target_output) - self.pred_output), -1)
        
        # update portion with mask over loss
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_to_update, tf.float32)

        # network update
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
