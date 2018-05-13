import random
import tensorflow as tf
import numpy as np
import pickle
import copy
from DRLP.utils import Settings
from DRLP.utils.dact import DiaAct
from DRLP.policy import PolicyUtils
from DRLP.policy.Policy import Policy, TerminalState, TerminalAction
from DRLP.policy.a2c.a2c import A2CNetwork
from DRLP.policy.replaybuffer.ReplayBufferEpisode import ReplayBufferEpisode


class A2CPolicy(Policy):

    def __init__(self, learning, in_policy_file, out_policy_file):
        super(A2CPolicy, self).__init__(learning)
        self.summary_act = None

        tf.reset_default_graph()

        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file

        self.random_seed = 1234
        self.learning_rate = 0.001
        self.tau = 0.001
        self.h1_size = 130
        self.h2_size = 130
        self.replay_type = 'vanilla'
        self.buffer_size = 1000
        self.batch_size = 32

        self.exploration_type = 'e-greedy'  # Boltzman
        self.training_frequency = 2
        self.epsilon = 1.0
        self.epsilon_start = 1.0

        self.importance_sampling = False
        self.gamma = 1.0
        self.use_alter = False  # add user reqalts intent into the belief state

        if Settings.config.has_option('general', 'seed'):
            self.random_seed = Settings.config.getint('general', 'seed')
        if Settings.config.has_option('dqnpolicy', 'learning_rate'):
            self.learning_rate = Settings.config.getfloat('dqnpolicy', 'learning_rate')
        if Settings.config.has_option('dqnpolicy', 'tau'):
            self.tau = Settings.config.getfloat('dqnpolicy', 'tau')
        if Settings.config.has_option('dqnpolicy', 'h1_size'):
            self.h1_size = Settings.config.getint('dqnpolicy', 'h1_size')
        if Settings.config.has_option('dqnpolicy', 'h2_size'):
            self.h2_size = Settings.config.getint('dqnpolicy', 'h2_size')
        if Settings.config.has_option('dqnpolicy', 'replay_type'):
            self.replay_type = Settings.config.get('dqnpolicy', 'replay_type')
        if Settings.config.has_option('dqnpolicy', 'buffer_size'):
            self.buffer_size = Settings.config.getint('dqnpolicy', 'buffer_size')
        if Settings.config.has_option('dqnpolicy', 'batch_size'):
            self.batch_size = Settings.config.getint('dqnpolicy', 'batch_size')

        if Settings.config.has_option('dqnpolicy', 'exploration_type'):
            self.exploration_type = Settings.config.get('dqnpolicy', 'exploration_type')
        if Settings.config.has_option('dqnpolicy', 'training_frequency'):
            self.training_frequency = Settings.config.getint('dqnpolicy', 'training_frequency')
        if Settings.config.has_option('dqnpolicy', 'epsilon'):
            self.epsilon = Settings.config.getfloat('dqnpolicy', 'epsilon')
        if Settings.config.has_option('dqnpolicy', 'epsilon_start'):
            self.epsilon_start = Settings.config.getfloat('dqnpolicy', 'epsilon_start')

        if Settings.config.has_option('dqnpolicy', 'importance_sampling'):
            self.importance_sampling = Settings.config.getboolean('dqnpolicy', 'importance_sampling')
        if Settings.config.has_option('dqnpolicy', 'gamma'):
            self.gamma = Settings.config.getfloat('dqnpolicy', 'gamma')

        if Settings.config.has_option('policy', 'use_alter'):
            self.use_alter = Settings.config.getboolean('policy', 'use_alter')

        self.state_dim = PolicyUtils.get_state_dim(self.use_alter)
        self.action_dim = len(self.actions.action_names)

        self.mu_prob = 0.  # behavioral policy

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            np.random.seed(self.random_seed)

            if self.replay_type == 'vanilla':
                self.episode = ReplayBufferEpisode(self.buffer_size, self.batch_size, self.random_seed)

            self.sample_count = 0
            self.episode_count = 0

            self.a2c = A2CNetwork(self.graph, self.sess, self.state_dim, self.action_dim, self.learning_rate, self.tau,
                                  self.h1_size, self.h2_size, self.learning)

            init = tf.global_variables_initializer()
            self.sess.run(init)

            self.load_policy(self.in_policy_file)
            print 'loaded replay size:', self.episode.size()

    def act_on(self, belief_state, entities):
        if self.last_system_action is None and self.start_with_hello:
            sys_act, next_action_idx = 'hello()', -1
        else:
            sys_act, next_action_idx = self.next_action(belief_state, entities)
        self.last_system_action = sys_act
        self.summary_act = next_action_idx
        self.prev_belief = belief_state

        sys_action = DiaAct(sys_act)
        return sys_action

    def record(self, reward, weight=None, state=None, action=None):
        if self.episode is None:
            if self.replay_type == 'vanilla':
                self.episode = ReplayBufferEpisode(self.buffer_size, self.batch_size, self.random_seed)

        if self.act_to_be_recorded is None:
            self.act_to_be_recorded = self.summary_act

        if state is None:
            state = self.prev_belief
        if action is None:
            action = self.act_to_be_recorded

        c_state, c_action = self.convert_state_action(state, action)

        # TODO:: it is need?
        # normalising total return to -1~1
        reward /= 20.0

        value = self.a2c.predict_value([c_state])
        policy_mu = self.mu_prob

        if self.replay_type == 'vanilla':
            self.episode.record(state=c_state, state_ori=state, action=c_action, reward=reward,
                                value=float(value[0][0]), distribution=policy_mu)

        self.act_to_be_recorded = None
        self.sample_count += 1

    def finalize_record(self, reward):
        if self.episode is None:
            print 'Error!! A2C Episode is not initialized!!'
            return

        # TODO:: it is need?
        # normalising total return to -1~1
        reward /= 20.0

        terminal_state, terminal_action = self.convert_state_action(TerminalState(), TerminalAction())
        value = 0.0     # not effect on experience replay

        if self.replay_type == 'vanilla':
            self.episode.record(state=terminal_state, state_ori=TerminalState(), action=terminal_action,
                                reward=reward, value=value, terminal=True, distribution=None)

    def convert_state_action(self, state, action):
        if isinstance(state, TerminalState):
            return [0] * self.state_dim, action
        else:
            flat_belief = PolicyUtils.flatten_belief(state, self.use_alter)
            return flat_belief, action

    def next_action(self, belief_state, entities):

        belief_vec = PolicyUtils.flatten_belief(belief_state, self.use_alter)
        exec_mask = self.actions.get_executable_mask(belief_state, entities)

        action_prob, value = self.a2c.predict_action_value(np.reshape(belief_vec, (1, len(belief_vec))))
        action_q_admissible = np.add(action_prob, np.array(exec_mask))

        next_action_idx = -1
        if self.exploration_type == 'e-greedy':
            greedy_next_action_idx = np.argmax(action_q_admissible)

            # epsilon greedy
            if self.learning and Settings.random.rand() < self.epsilon:
                admissible = [i for i, x in enumerate(exec_mask) if x == 0.0]
                random.shuffle(admissible)
                next_action_idx = admissible[0]

                # Importance sampling
                if next_action_idx == greedy_next_action_idx:
                    self.mu_prob = self.epsilon / float(self.action_dim) + 1 - self.epsilon
                else:
                    self.mu_prob = self.epsilon / float(self.action_dim)
            else:
                next_action_idx = greedy_next_action_idx
                self.mu_prob = self.epsilon / float(self.action_dim) + 1 - self.epsilon

        summary_act = self.actions.action_names[next_action_idx]
        master_act = self.actions.convert(belief_state, summary_act, entities)

        return master_act, next_action_idx

    def restart(self):
        self.last_system_action = None
        self.prev_belief = None
        self.act_to_be_recorded = None
        self.summary_act = None
        self.epsilon = self.epsilon_start

    def save_policy(self):
        if self.learning:
            self.a2c.save_network(self.out_policy_file+'.a2c')
            f = open(self.out_policy_file + '.episode', 'wb')
            for obj in [self.sample_count, self.episode]:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    def load_policy(self, filename):
        self.a2c.load_network(filename + '.a2c')
        try:
            f = open(filename + '.episode', 'rb')
            loaded_objects = []
            for i in range(2):  # load nn params and collected data
                loaded_objects.append(pickle.load(f))
            self.sample_count = int(loaded_objects[0])
            self.episode = copy.deepcopy(loaded_objects[1])
            f.close()
        except IOError:
            print 'loading only models...'

    def train(self):
        """
        Call this function when the episode ends
        """
        self.episode_count += 1

        if self.sample_count >= self.batch_size * 3 and self.episode_count % self.training_frequency == 0:
            s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, v_batch, mu_policy = \
                self.episode.sample_batch()

            discounted_r_batch = []
            advantage_batch = []

            weights, rho_forward, rho_whole, r_new = self._weights_importance_sampling(mu_policy, s_batch,
                                                                                       a_batch, r_batch)

            weights = np.nan_to_num(weights)
            rho_forward = np.nan_to_num(rho_forward)

            if self.replay_type == 'vanilla':
                for item_r, item_v in zip(r_new, v_batch):
                    rlist = []
                    for idx in range(len(item_r)):
                        r = self._calculate_discountR(item_r, idx, rho_forward)
                        rlist.append(r)

                    a = self._calculate_advantage(item_r, item_v)

                    # flatten nested numpy array and turn it into list
                    discounted_r_batch += rlist
                    advantage_batch += a.tolist()

            # change index-based a_batch to one-hot-based a_batch
            a_batch_one_hot = np.eye(self.action_dim)[np.concatenate(a_batch, axis=0).tolist()]

            if self.importance_sampling:
                discounted_r_batch = np.clip(discounted_r_batch, -2, 2)

            value_loss, policy_loss, entropy, all_loss, optimise = \
                self.a2c.train(np.concatenate(np.array(s_batch), axis=0).tolist(), a_batch_one_hot,
                               discounted_r_batch, advantage_batch, weights, rho_forward)

    def _weights_importance_sampling(self, mu_policy, s_batch, a_batch, r_batch):
        """
        Weights for importance sampling
        it goes through each dialogue and computes in reverse order cumulative prod:
        # rho_n = pi_n / mu_n
        # ...
        # rho_1 = pi_1 / mu_1 *  ... * pi_n / mu_n
        # using dialogue and weight_cum lists

        :param mu_policy:
        :param s_batch:
        :param a_batch:
        :param r_batch:
        :return:
        """

        mu_policy = np.asarray(mu_policy)
        mu_cum = []
        lengths = []  # to properly divde on dialogues pi_policy later on
        for mu in mu_policy:
            lengths.append(len(mu))
            mu = np.asarray(mu).astype(np.longdouble)
            mu_cum.append(np.cumprod(mu[::-1])[::-1])  # going forward with cumulative product

        mu_policy = np.concatenate(np.array(mu_policy), axis=0).tolist()  # concatenate all behavioral probs
        lengths = np.cumsum(lengths)  # time steps for ends of dialogues

        pi_policy = self.a2c.get_policy(np.concatenate(np.array(s_batch), axis=0).tolist())[0]  # policy given s_t
        columns = np.asarray([np.concatenate(a_batch, axis=0).tolist()]).astype(int)  # actions taken at s_t
        rows = np.asarray([ii for ii in range(len(pi_policy))])
        pi_policy = pi_policy[rows, columns][0].astype(np.longdouble)

        rho_forward = []  # rho_forward from eq. 3.3 (the first one)
        rho_whole = []  # product across the whole dialogue from eq. 3.3 (the second one)

        # Precup version
        r_vector = np.concatenate(np.array(r_batch), axis=0).tolist()
        r_weighted = []

        for ii in range(len(lengths) - 1):  # over dialogues
            weight_cum = 1.
            dialogue = []

            # first case
            if ii == 0:
                for pi, mu in zip(pi_policy[0:lengths[0]], mu_policy[0:lengths[0]]):
                    weight_cum *= pi / mu
                    dialogue.append(weight_cum)

                dialogue = np.array(dialogue)

                if self.importance_sampling:
                    dialogue = np.clip(dialogue, -1, 1)
                else:
                    dialogue = np.ones(dialogue.shape)
                dialogue = dialogue.tolist()

                rho_forward.extend(dialogue)
                rho_whole.extend(np.ones(len(dialogue)) * dialogue[-1])
                r_weighted.extend(r_vector[0:lengths[0]] * np.asarray(dialogue))
                dialogue = []

            for pi, mu in zip(pi_policy[lengths[ii]:lengths[ii + 1]], mu_policy[lengths[ii]:lengths[ii + 1]]):
                weight_cum *= pi / mu
                dialogue.append(weight_cum)

            dialogue = np.array(dialogue)
            if self.importance_sampling:
                dialogue = np.clip(dialogue, -1, 1)
            else:
                dialogue = np.ones(dialogue.shape)
            dialogue = dialogue.tolist()

            rho_forward.extend(dialogue)
            rho_whole.extend(np.ones(len(dialogue)) * dialogue[-1])
            r_weighted.extend(r_vector[lengths[ii]: lengths[ii + 1]] * np.asarray(dialogue))

        # go back to original form:
        ind = 0
        r_new = copy.deepcopy(r_batch)
        for id, batch in enumerate(r_new):
            for id2, _ in enumerate(batch):
                r_new[id][id2] = r_weighted[ind]
                ind += 1

        # ONE STEP WEIGHTS
        weights = np.asarray(pi_policy) / np.asarray(mu_policy)
        if self.importance_sampling:
            weights = np.clip(weights, -1, 1)
        else:
            weights = np.ones(weights.shape)

        return weights, rho_forward, rho_whole, r_new

    def _calculate_discountR(self, r_episode, idx, rho_forward):
        """
        Here we take the rewards and values from the rolloutv, and use them to
        generate the advantage and discounted returns.
        The advantage function uses "Generalized Advantage Estimation"

        :param r_episode:
        :param idx:
        :param rho_forward:
        :return:
        """
        bootstrap_value = 0.0
        # r_episode rescale by rhos?
        r_episode_plus = np.asarray(r_episode[idx:] + [bootstrap_value])
        if self.importance_sampling:
            r_episode_plus = r_episode_plus
        else:
            r_episode_plus = r_episode_plus / rho_forward[idx]
        discounted_r_episode = PolicyUtils.discount(r_episode_plus, self.gamma)[:-1]

        return discounted_r_episode[0]

    def _calculate_advantage(self, r_episode, v_episode):
        """
        Here we take the rewards and values from the rolloutv, and use them to
        generate the advantage and discounted returns.
        The advantage function uses "Generalized Advantage Estimation"
        :param r_episode:
        :param v_episode:
        :return:
        """

        bootstrap_value = 0.0
        v_episode_plus = np.asarray(v_episode + [bootstrap_value])
        # change sth here
        advantage = r_episode + self.gamma * v_episode_plus[1:] - v_episode_plus[:-1]
        advantage = PolicyUtils.discount(advantage, self.gamma)

        return advantage
