import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class A2CNetwork(object):

    def __init__(self, graph, sess, state_dim, action_dim, learning_rate, h1_size=130, h2_size=50, learning=True):
        self.graph = graph
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.learning = learning

        self.inputs  = tf.placeholder(tf.float32, [None, self.s_dim])
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim])

        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, self.h1_size], stddev=0.01), name='a2c_W_fc1')
        b_fc1 = tf.Variable(tf.zeros([self.h1_size]), name='a2c_b_fc1')
        h_fc1 = tf.nn.relu6(tf.matmul(self.inputs, W_fc1) + b_fc1)

        if self.h2_size > 0:
            # value function
            W_value = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01), name='a2c_W_value_1')
            b_value = tf.Variable(tf.zeros([self.h2_size]), name='a2c_b_value_1')
            h_value = tf.nn.relu6(tf.matmul(h_fc1, W_value) + b_value)

            W_value = tf.Variable(tf.truncated_normal([self.h2_size, 1], stddev=0.01), name='a2c_W_value_2')
            b_value = tf.Variable(tf.zeros([1]), name='a2c_b_value2')
            self.value = tf.matmul(h_value, W_value) + b_value

            # policy function
            W_policy = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01), name='a2c_W_policy_1')
            b_policy = tf.Variable(tf.zeros([self.h2_size]), name='a2c_b_policy_1')
            h_policy = tf.nn.relu6(tf.matmul(h_fc1, W_policy) + b_policy)

            W_policy = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01), name='a2c_W_policy_2')
            b_policy = tf.Variable(tf.zeros([self.a_dim]), name='a2c_b_policy2')
            self.policy = tf.nn.softmax(tf.matmul(h_policy, W_policy) + b_policy) + 0.00001

        else:   # 1 hidden layer
            # value function
            W_value = tf.Variable(tf.truncated_normal([self.h1_size, 1], stddev=0.01), name='a2c_W_value_3')
            b_value = tf.Variable(tf.zeros([1]), name='a2c_b_value_3')
            self.value = tf.matmul(h_fc1, W_value) + b_value

            # policy function
            W_policy = tf.Variable(tf.truncated_normal([self.h1_size, self.a_dim], stddev=0.01))
            b_policy = tf.Variable(tf.zeros([self.a_dim]))
            self.policy = tf.nn.softmax(tf.matmul(h_fc1, W_policy) + b_policy) + 0.00001

        # all parameters
        self.vars = tf.trainable_variables()

        #######################################
        # Reinforcement Learning
        #######################################
        # Only the worker network need ops for loss functions and gradient updating.
        self.actions_one_hot = self.actions
        self.target_v = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.weights = tf.placeholder(tf.float32, [None])
        self.rho_forward = tf.placeholder(tf.float32, [None])

        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_one_hot, [1])

        # loss function
        self.value_diff = self.rho_forward * tf.square(self.target_v - tf.reshape(self.value, [-1]))
        self.value_diff = tf.clip_by_value(self.value_diff, -2, 2)
        self.value_loss = 0.5 * tf.reduce_sum(self.value_diff)

        self.policy_diff = tf.log(self.responsible_outputs) * self.advantages * self.weights
        self.policy_diff = tf.clip_by_value(self.policy_diff, -20, 20)
        self.policy_loss = -tf.reduce_sum(self.policy_diff)

        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))

        self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.1

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # add clipping too!
        # clipping
        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs]
        self.optimize = self.optimizer.apply_gradients(capped_gvs)

        #######################################
        # Supervised Learning
        #######################################
        self.policy_y = tf.placeholder(tf.int64, [None])
        self.policy_y_one_hot = tf.one_hot(self.policy_y, self.a_dim, 1.0, 0.0, name='a2c_policy_y_one_hot')

        self.loss_sl = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy, labels=self.policy_y_one_hot))

        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars
                                if 'bias' not in v.name]) * 0.001

        self.loss_combined = self.loss_sl + self.lossL2

        self.optimizer_sl = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize_sl = self.optimizer_sl.minimize(self.loss_combined)

        self.policy_picked = tf.argmax(self.policy, 1)

        correct_prediction = tf.equal(tf.argmax(self.policy, 1), self.policy_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.params = [v.name for v in tf.trainable_variables()]

        ###################################################
        # Supervised Learning + Reinforcement Learning
        ###################################################
        self.loss_all = self.loss + self.loss_sl

        self.optimizer_all = tf.train.AdamOptimizer(self.learning_rate)

        # clipping
        gvs = self.optimizer_all.compute_gradients(self.loss_all)
        capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs]
        self.optimize_all = self.optimizer.apply_gradients(capped_gvs)

    def get_policy(self, inputs):
        return self.sess.run([self.policy], feed_dict={
            self.inputs: inputs
        })

    def get_loss(self, inputs, actions, discounted_rewards, advantages, weights, rho_forward):
        return self.sess.run([self.value_loss, self.policy_loss, self.entropy, self.loss],
                             feed_dict={
                                 self.inputs: inputs,
                                 self.actions: actions,
                                 self.target_v: discounted_rewards,
                                 self.advantages: advantages,
                                 self.weights: weights,
                                 self.rho_forward: rho_forward
                             })

    def train(self, inputs, actions, discounted_rewards, advantages, weights, rho_forward):
        return self.sess.run([self.value_loss, self.policy_loss, self.entropy, self.loss, self.optimize],
                             feed_dict={
                                 self.inputs: inputs,
                                 self.actions: actions,
                                 self.target_v: discounted_rewards,
                                 self.advantages: advantages,
                                 self.weights: weights,
                                 self.rho_forward: rho_forward
                             })

    def predict_action_value(self, inputs):
        return self.sess.run([self.policy, self.value], feed_dict={
            self.inputs: inputs
        })

    def predict_value(self, inputs):
        return self.sess.run([self.value], feed_dict={
            self.inputs: inputs
        })

    def load_network(self, filename):
        with self.graph.as_default():
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, filename)
                print "\nSuccessfully loaded:", filename
            except:
                print "Could not find old network weights"

    def save_network(self, filename):
        print 'save a2c-network in:', filename
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, filename)
