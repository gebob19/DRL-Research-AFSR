import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np 
from utils import Network

class DensityModel(object):
    def __init__(self, graph_args):
        # unpacking
        self.ob_dim = graph_args['ob_dim']
        self.learning_rate = graph_args['learning_rate']
        self.z_size = graph_args['z_size']
        self.kl_weight = graph_args['kl_weight']
        # network operations params
        self.hid_size = graph_args['hid_size']
        self.n_hidden = graph_args['n_hidden']
        self.bonus_multiplier = graph_args['bonus_multiplier']
        
        self.state1, self.state2 = self.define_placeholders()
        # q(z_1 | s_1), q(z_2 | s_2), p(z), p(y | z)
        self.encoder1, self.encoder2, self.prior, self.discriminator = self.forward_pass(self.state1, self.state2)
        self.discrim_target = tf.placeholder(shape=[None, 1], name="discrim_target", dtype=tf.int32)

        self.log_likelihood = tf.squeeze(self.discriminator.log_prob(self.discrim_target), axis=1)
        self.likelihood = tf.squeeze(self.discriminator.prob(self.discrim_target), axis=1)
        
        self.kl = self.encoder1.kl_divergence(self.prior) + self.encoder2.kl_divergence(self.prior)

        assert len(self.log_likelihood.shape) == len(self.likelihood.shape) == len(self.kl.shape) == 1
        
        self.elbo = tf.reduce_mean(self.log_likelihood - self.kl_weight * self.kl)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.elbo)

        # denfine another graph with reused components
        # used to calculate the bonus of states which have already been encoded
        self.set_encoded_graph()


    def define_placeholders(self):
        state1 = tf.placeholder(shape=((None,) + (self.ob_dim)), name="s1", dtype=tf.float32)
        state2 = tf.placeholder(shape=((None,) + (self.ob_dim)), name="s2", dtype=tf.float32)
        return state1, state2
    
    def set_session(self, sess):
        self.sess = sess

    #Network(input_tensor, output_size, scope, fsize, conv_depth, n_hidden_dense=0, activation=tf.tanh, output_activation=None):
    def make_encoder(self, state, z_size, scope):
        """ Encodes the given state to z_size => create guass. distribution for q(z | s)
        """
        # conv operations
        z_mean = Network(state, z_size, scope, self.hid_size, conv_depth=self.n_hidden)
        z_logstd = tf.get_variable("logstd", shape=(z_size,)) 
        return tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.exp(z_logstd))

    def make_prior(self, z_size):
        """ Create Prior to map states too => p(z), we will use a normal guass distrib
        """
        prior_mean = tf.zeros((z_size,))
        prior_logstd = tf.zeros((z_size,))
        return tfp.distributions.MultivariateNormalDiag(loc=prior_mean, scale_diag=tf.exp(prior_logstd))

    def make_discriminator(self, z, output_size, scope, n_layers, hid_size, reuse=False):
        """ Predict D(z = [z1, z2]) => p(y | z)
        """
        logit = Network(z, output_size, scope, hid_size, conv_depth=0, n_hidden_dense=n_layers, reuse=reuse)
        return tfp.distributions.Bernoulli(logit)

    def forward_pass(self, state1, state2):
        # Reuse
        make_encoder1 = tf.make_template('encoder1', self.make_encoder)
        make_encoder2 = tf.make_template('encoder2', self.make_encoder)
        self.discrim_template = tf.make_template('decoder', self.make_discriminator)

        # Encoder
        encoder1 = make_encoder1(state1, self.z_size, 'z1')
        encoder2 = make_encoder2(state2, self.z_size, 'z2')

        # Prior
        prior = self.make_prior(self.z_size)

        # Sampled Latent (some noise)
        self.z1 = encoder1.sample()
        z2 = encoder2.sample()
        z = tf.concat([self.z1, z2], axis=1)

        # Discriminator
        discriminator = self.discrim_template(z, 1, 'discriminator', self.n_hidden, self.hid_size)
        return encoder1, encoder2, prior, discriminator

    def set_encoded_graph(self):
        self.enc1, self.enc2 = tf.placeholder(shape=(None, self.z_size), dtype=tf.float32), tf.placeholder(shape=(None, self.z_size), dtype=tf.float32)
        enc = tf.concat([self.enc1, self.enc2], axis=1)
        discrim = self.discrim_template(enc, 1, 'discriminator', self.n_hidden, self.hid_size, reuse=True)
        self.enc_likelihood = tf.squeeze(discrim.prob(self.discrim_target), axis=1)


    def update(self, state1, state2, target):
        _, ll, kl, elbo = self.sess.run([self.update_op, self.log_likelihood, self.kl, self.elbo], feed_dict={
            self.state1: state1,
            self.state2: state2,
            self.discrim_target: target
        })
        return ll, kl, elbo
    
    def get_encoding(self, states):
        """Assuming only encode a single state at a time
        We will call this to use in our state dynamics fcn
        """
        return self.sess.run(self.z1, feed_dict={
            self.state1: states
        })

    def get_likelihood(self, state1, state2, encoded=False):
        # set target
        bs, _ = state1.shape
        target = np.zeros((bs, 1))
        for i, (s1, s2) in enumerate(zip(state1, state2)):
            if s1.all() == s2.all(): target[i] = [1]
        # select graph to execute
        if not encoded:
            likelihood = self.sess.run(self.likelihood, feed_dict={
                self.state1: state1,
                self.state2: state2,
                self.discrim_target: target
            })
        else:
            likelihood = self.sess.run(self.enc_likelihood, feed_dict={
                self.enc1: state1,
                self.enc2: state2,
                self.discrim_target: target
            })
        return likelihood

    def get_prob(self, state, encoded=False):
        likelihood = self.get_likelihood(state, state, encoded=encoded)
        # avoid divide by 0 and log(0)
        likelihood = np.clip(np.squeeze(likelihood), 1e-5, 1-1e-5)
        prob = (1 - likelihood) / likelihood
        return prob

    def get_bonus(self, states, evaluate=False, encoded=False):
        if evaluate: states = states.eval()
        probs = self.get_prob(states, encoded=encoded)
        return -np.log(probs)
    
    def modify_reward(self, state, rewards):
        bonus = self.get_bonus(state)
        return rewards + self.bonus_multiplier * bonus