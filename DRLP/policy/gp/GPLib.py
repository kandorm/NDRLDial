import os
import math
import scipy.stats
import pickle as pkl
import numpy as np
from DRLP.utils import utils
from DRLP.policy.Policy import State, Action, TerminalState, TerminalAction
from DRLP.utils import Settings
from DRLP.policy import PolicyUtils


class LearnerInterface(object):

    def __init__(self):
        self._inputDictFile = ""
        self._inputParamFile = ""
        self._outputDictFile = ""
        self._outputParamFile = ""
        self.initial = False
        self.terminal = False

    def policy(self, state, kernel=None, executable=None):
        pass

    def save_policy(self):
        pass

    def read_policy(self):
        pass

    def learning_step(self, prev_state, prev_action, reward, cur_state, cur_action, kernel):
        pass


class GPSARSAPrior(LearnerInterface):

    def __init__(self, in_policy_file, out_policy_file, num_prior=-1, learning=False):
        super(GPSARSAPrior, self).__init__()

        self._inPolicyFile = in_policy_file
        self._outPolicyFile = out_policy_file
        self._inputDictFile = in_policy_file + "." + str(num_prior) + ".prior.dct"
        self._inputParamFile = in_policy_file + "." + str(num_prior) + ".prior.prm"
        self._outputDictFile = out_policy_file + "." + str(num_prior) + ".prior.dct"
        self._outputParamFile = out_policy_file + "." + str(num_prior) + ".prior.prm"

        self.learning = learning
        self._prior = None
        self._superior = None
        self._random = False

        self.params = {'_dictionary': [], '_alpha_tilda': []}

        if num_prior > 0:
            self._prior = GPSARSAPrior(in_policy_file, out_policy_file, num_prior - 1)
            self._prior._superior = self

        if self._prior:
            self.read_policy()

    def QvalueMean(self, state, action, kernel):
        """
        :returns: Q value mean for a given state, action and the kernel function\
        Recursively calculates the mean given depending on the number of recursive priors
        """
        q_prior = 0
        q_val = 0
        if self._prior is not None:
            q_prior = self._prior.QvalueMean(state, action, kernel)

        if len(self.params['_dictionary']) > 0:
            k_tilda_t = self._k_tilda(state, action, kernel)
            q_val = np.dot(self.params['_alpha_tilda'], k_tilda_t)

        mean = q_prior + q_val

        return mean

    def read_policy(self):
        """
        Reads dictionary file and parameter file
        """
        if not os.path.isfile(self._inputDictFile) or not os.path.isfile(self._inputParamFile):
            print 'Warning: GPSARSAPrior->inPolicyFile:', self._inPolicyFile, 'does not exist!!'
        else:
            self._read_dictionary()
            self._read_parameters()

    def _k_tilda(self, state, action, kernel):
        """
        Based on product between action and belief kernels.
        O(N) in dictionary size N.
        :returns:vector of kernel values of given state, action and kernel with all state-action pairs in the dictionary
        """
        res = []
        for [d_state, d_action] in self.params['_dictionary']:
            ker_act = kernel.action_kernel(action, d_action)
            if ker_act > 0:
                if self._prior is not None:
                    ker_state = kernel.prior_kernel(state, d_state)
                else:
                    ker_state = kernel.belief_kernel(state, d_state)

                res.append(ker_act * ker_state)
            else:
                res.append(ker_act)
        return np.array(res)

    def _read_dictionary(self):
        """
        Reads input policy dictionary file
        """
        with open(self._inputDictFile, 'rb') as pkl_file:
            self.params['_dictionary'] = pkl.load(pkl_file)

    def _read_parameters(self):
        """
        Reads input policy parameters
        """
        with open(self._inputParamFile, 'rb') as pkl_file:
            self.params['_alpha_tilda'] = pkl.load(pkl_file)


class GPSARSA(GPSARSAPrior):

    def __init__(self, in_policy_file, out_policy_file, learning=False):
        super(GPSARSA, self).__init__(in_policy_file, out_policy_file, learning=learning)

        self._inputDictFile = in_policy_file + ".dct"
        self._inputParamFile = in_policy_file + ".prm"
        self._outputDictFile = out_policy_file + ".dct"
        self._outputParamFile = out_policy_file + ".prm"

        # for policy training
        self.save_as_prior = False
        self.initial = False
        self.terminal = False

        # self.learning = learning => set in GPSARSAPrior __init__
        self._prior = None
        self._random = False
        self.numpyFileFormat = False
        self._scale = 3
        self._num_prior = 0

        self._gamma = 1.0
        self._sigma = 5.0
        self._nu = 0.001

        self.params = {'_K_tilda_inv': np.zeros((1, 1)),
                       '_C_tilda': np.zeros((1, 1)),
                       '_c_tilda': np.zeros(1),
                       '_a': np.ones(1),
                       '_alpha_tilda': np.zeros(1),
                       '_dictionary': [],
                       '_d': 0,
                       '_s': float('inf')}

        if Settings.config.has_option('gpsarsa', "saveasprior"):
            self.save_as_prior = Settings.config.getboolean('gpsarsa', "saveasprior")
        if Settings.config.has_option("gpsarsa", "random"):
            self._random = Settings.config.getboolean('gpsarsa', "random")
        if Settings.config.has_option("gpsarsa", "saveasnpy"):
            self.numpyFileFormat = Settings.config.getboolean('gpsarsa', "saveasnpy")
        if Settings.config.has_option('gpsarsa', "scale"):
            self._scale = Settings.config.getint('gpsarsa', "scale")
        if Settings.config.has_option('gpsarsa', "numprior"):
            self._num_prior = Settings.config.getint('gpsarsa', "numprior")
        if Settings.config.has_option('gpsarsa', "gamma"):
            self._gamma = Settings.config.getfloat('gpsarsa', "gamma")
        if Settings.config.has_option('gpsarsa', "sigma"):
            self._sigma = Settings.config.getfloat('gpsarsa', "sigma")
        if Settings.config.has_option('gpsarsa', "nu"):
            self._nu = Settings.config.getfloat('gpsarsa', "nu")

        # load the parameters
        self.read_policy()

        if self._num_prior > 0:
            self._prior = GPSARSAPrior(in_policy_file, '', self._num_prior - 1)
            self._prior._superior = self

    def QvalueMeanVar(self, state, action, kernel):
        """
        Gets mean and variance of Q-value at (S,A) pair for given kernel
        :param: belief_state
        :param: action
        :param: kernel
        :returns: mean and variance of GP for given belief state, action and kernel
        """
        q_prior = 0
        q_val = 0
        q_var = 0
        if self._prior is not None:
            q_prior = self._prior.QvalueMean(state, action, kernel)

        if len(self.params['_dictionary']) > 0:
            k_tilda_t = self._k_tilda(state, action, kernel)
            q_val = np.dot(k_tilda_t, self.params['_alpha_tilda'])
            q_var = np.dot(k_tilda_t, np.dot(self.params['_C_tilda'], k_tilda_t))

        mean = q_prior + q_val
        if self._prior is not None:
            q_org = kernel.prior_kernel(state, state) * kernel.action_kernel(action, action)
        else:
            q_org = kernel.belief_kernel(state, state) * kernel.action_kernel(action, action)
        var = q_org - q_var

        if var < 0:
            var = 0

        return [mean, var]

    def policy(self, state, kernel=None, executable=None):
        """
        :param state: GPState
        :type state: GPState
        :param kernel: kernel
        :type kernel: Kernel
        :param executable: executable action space
        :type executable: list
        :returns: best action according to sampled Q values
        """
        if executable is None or len(executable) == 0:
            exit('Error!! No executable actions!!!')
            return None

        if self._random:
            return [Settings.random.choice(executable).act, 0, 0]

        Q = []
        for action in executable:
            [mean, var] = self.QvalueMeanVar(state, action, kernel)
            if self._scale <= 0:
                gauss_var = 0
                value = mean
            else:
                gauss_var = self._scale * math.sqrt(var)
                value = gauss_var * Settings.random.randn() + mean
            Q.append((action, value, mean, gauss_var))
        Q = sorted(Q, key=lambda q_value: q_value[1], reverse=True)

        best_action, best_action_sampled_q_value = Q[0][0], Q[0][1]
        action_likelihood = 0
        if Q[0][3] != 0:
            action_likelihood = scipy.stats.norm(Q[0][2], Q[0][3]).pdf(best_action_sampled_q_value)
        return [best_action, best_action_sampled_q_value, action_likelihood]

    def save_policy(self):
        """
        Saves the GP dictionary (.dct) and parameters (.prm). Saves as a prior if self.save_as_prior is True.
        :returns: None
        """
        out_policy_file = self._outPolicyFile
        if self.save_as_prior:
            prior_dict_file = out_policy_file + "." + str(self._num_prior) + ".prior.dct"
            prior_param_file = out_policy_file + "." + str(self._num_prior) + ".prior.prm"
            self._save_prior(prior_dict_file, prior_param_file)
        else:
            self._save_dictionary()
            self._save_parameters()

    def _save_prior(self, prior_dict_file, prior_param_file):
        """
        Saves the current GP as a prior (these are only the parameters needed to estimate the mean)
        """
        utils.check_dir_exists_and_make(prior_dict_file)
        with open(prior_dict_file, 'wb') as pkl_file:
            pkl.dump(self.params['_dictionary'], pkl_file)
        utils.check_dir_exists_and_make(prior_param_file)
        with open(prior_param_file, 'wb') as pkl_file:
            pkl.dump(self.params['_alpha_tilda'], pkl_file)

    def _save_dictionary(self):
        """
        Saves dictionary
        :param None:
        :returns None:
        """
        output_dict_file = self._outputDictFile
        utils.check_dir_exists_and_make(output_dict_file)
        with open(output_dict_file, 'wb') as pkl_file:
            pkl.dump(self.params['_dictionary'], pkl_file)

    def _save_parameters(self):
        """
        Save parameter file
        """
        output_param_file = self._outputParamFile
        utils.check_dir_exists_and_make(output_param_file)
        with open(output_param_file, 'wb') as pkl_file:
            if self.numpyFileFormat:
                np.savez(pkl_file, _K_tilda_inv=self.params['_K_tilda_inv'], _C_tilda=self.params['_C_tilda'],
                         _c_tilda=self.params['_c_tilda'], _a=self.params['_a'],
                         _alpha_tilda=self.params['_alpha_tilda'], _d=self.params['_d'], _s=self.params['_s'])
            else:
                # ORDER MUST BE THE SAME HERE AS IN readParameters() above.
                pkl.dump(self.params['_K_tilda_inv'], pkl_file)
                pkl.dump(self.params['_C_tilda'], pkl_file)
                pkl.dump(self.params['_c_tilda'], pkl_file)
                pkl.dump(self.params['_a'], pkl_file)
                pkl.dump(self.params['_alpha_tilda'], pkl_file)
                pkl.dump(self.params['_d'], pkl_file)
                pkl.dump(self.params['_s'], pkl_file)

    def read_policy(self):
        """
        Reads dictionary and parameter file
        """
        if not os.path.isfile(self._inputDictFile) or not os.path.isfile(self._inputParamFile):
            print 'Warning: GPSARSA->inPolicyFile:', self._inPolicyFile, 'does not exist!!'
        else:
            self._read_dictionary()
            self._read_parameters()

    def _read_dictionary(self):
        """
        Reads dictionary
        """
        input_dict_file = self._inputDictFile
        if input_dict_file not in ["", ".dct"]:
            with open(input_dict_file, 'rb') as pkl_file:
                self.params['_dictionary'] = pkl.load(pkl_file)
        else:
            print "GPSARSA dictionary file not given"

    def _read_parameters(self):
        """
        Reads parameter file
        """
        with open(self._inputParamFile, 'rb') as pkl_file:
            if self.numpyFileFormat:
                npzfile = np.load(pkl_file)
                try:
                    self.params['_K_tilda_inv'] = npzfile['_K_tilda_inv']
                    self.params['_C_tilda'] = npzfile['_C_tilda']
                    self.params['_c_tilda'] = npzfile['_c_tilda']
                    self.params['_a'] = npzfile['_a']
                    self.params['_alpha_tilda'] = npzfile['_alpha_tilda']
                    self.params['_d'] = npzfile['_d']
                    self.params['_s'] = npzfile['_s']
                except Exception as e:
                    print npzfile.files
                    raise e
            else:
                # ORDER MUST BE THE SAME HERE AS WRITTEN IN saveParameters() below.
                self.params['_K_tilda_inv'] = pkl.load(pkl_file)
                self.params['_C_tilda'] = pkl.load(pkl_file)
                self.params['_c_tilda'] = pkl.load(pkl_file)
                self.params['_a'] = pkl.load(pkl_file)
                self.params['_alpha_tilda'] = pkl.load(pkl_file)
                self.params['_d'] = pkl.load(pkl_file)
                self.params['_s'] = pkl.load(pkl_file)

    def learning_step(self, prev_state, prev_action, reward, cur_state, cur_action, kernel):
        """
        The main function of the GPSarsa algorithm
        :parameter:
        prev_state previous state
        prev_action previous action
        reward current reward
        cur_state next state
        cur_action next action
        kernel the kernel function

        Computes sufficient statistics needed to estimate the posterior of the mean and the covariance of the Gaussian process

        If the estimate of mean can take into account prior if specified
        """
        if self._prior is not None:
            if not self.terminal:
                offset = self._prior.QvalueMean(prev_state, prev_action, kernel) - \
                         self._gamma * self._prior.QvalueMean(cur_state, cur_action, kernel)
            else:
                offset = self._prior.QvalueMean(prev_state, prev_action, kernel)

            reward = reward - offset
        # INIT:
        if len(self.params['_dictionary']) == 0:
            self.params['_K_tilda_inv'] = np.zeros((1, 1))
            if self._prior is not None:
                self.params['_K_tilda_inv'][0][0] = 1.0 / (
                        kernel.prior_kernel(prev_state, prev_state) * kernel.action_kernel(prev_action, prev_action))
            else:
                self.params['_K_tilda_inv'][0][0] = 1.0 / (
                        kernel.belief_kernel(prev_state, prev_state) * kernel.action_kernel(prev_action, prev_action))

            self.params['_dictionary'].append([prev_state, prev_action])

        elif self.initial:
            k_tilda_prev = self._k_tilda(prev_state, prev_action, kernel)
            self.params['_a'] = np.dot(self.params['_K_tilda_inv'], k_tilda_prev)
            self.params['_c_tilda'] = np.zeros(len(self.params['_dictionary']))
            if self._prior is not None:
                delta_prev = kernel.prior_kernel(prev_state, prev_state) \
                             * kernel.action_kernel(prev_action, prev_action) - np.dot(k_tilda_prev, self.params['_a'])
            else:
                delta_prev = kernel.belief_kernel(prev_state, prev_state) \
                             * kernel.action_kernel(prev_action, prev_action) - np.dot(k_tilda_prev, self.params['_a'])

            self.params['_d'] = 0.0
            self.params['_s'] = float('inf')

            if delta_prev > self._nu:
                self._extend(delta_prev, prev_state, prev_action)

        k_tilda_prev = self._k_tilda(prev_state, prev_action, kernel)

        if self.terminal:
            _a_new = np.zeros(len(self.params['_dictionary']))
            delta_new = 0.0
            delta_k_tilda_new = k_tilda_prev
        else:
            k_tilda_new = self._k_tilda(cur_state, cur_action, kernel)
            _a_new = np.dot(self.params['_K_tilda_inv'], k_tilda_new)

            if self._prior is not None:
                curr_ker = kernel.prior_kernel(cur_state, cur_state) * kernel.action_kernel(cur_action, cur_action)
            else:
                curr_ker = kernel.belief_kernel(cur_state, cur_state) * kernel.action_kernel(cur_action, cur_action)

            ker_est = np.dot(k_tilda_new, _a_new)
            delta_new = curr_ker - ker_est
            delta_k_tilda_new = k_tilda_prev - self._gamma * k_tilda_new

        _d_new = reward + (
            0.0 if self.initial else (self._gamma * (self._sigma ** 2) * self.params['_d']) / self.params['_s']) \
            - np.dot(delta_k_tilda_new, self.params['_alpha_tilda'])

        self.params['_d'] = _d_new

        if delta_new < 0 and math.fabs(delta_new) > 0.0001:
            print "Negative sparcification " + str(delta_new)

        if delta_new > self._nu:
            self._extend_new(delta_new, cur_state, cur_action, kernel, _a_new, k_tilda_prev, k_tilda_new, delta_k_tilda_new)
        else:
            self._no_extend(_a_new, delta_k_tilda_new)

        self.params['_alpha_tilda'] += self.params['_c_tilda'] * (self.params['_d'] / self.params['_s'])
        self.params['_C_tilda'] += np.outer(self.params['_c_tilda'], self.params['_c_tilda']) / self.params['_s']

    def _extend(self, delta_prev, prev_state, prev_action):
        """
        Add points prev_state and prev_action in the dictionary and extend sufficient statistics matrices and vectors
        for one dimension
        Only used for the first state action pair in the episode
        """
        _a_prev = np.zeros(len(self.params['_dictionary']) + 1)
        _a_prev[-1] = 1.0
        _c_tilda_prev = np.zeros(len(self.params['_dictionary']) + 1)
        _K_tilda_inv_prev = self._extend_ktildainv(self.params['_K_tilda_inv'], self.params['_a'], delta_prev)
        _alpha_tilda_prev = self._extend_vector(self.params['_alpha_tilda'])
        _C_tilda_prev = self._extend_matrix(self.params['_C_tilda'])

        self.params['_a'] = _a_prev
        self.params['_alpha_tilda'] = _alpha_tilda_prev
        self.params['_c_tilda'] = _c_tilda_prev
        self.params['_C_tilda'] = _C_tilda_prev
        self.params['_K_tilda_inv'] = _K_tilda_inv_prev

        self.params['_dictionary'].append([prev_state, prev_action])

    def _extend_matrix(self, matrix):
        """
        Extend the dimentionality of matrix by one row and column -- new elements are zeros.
        """
        len_m = len(matrix[0])
        m_new = np.zeros((len_m + 1, len_m + 1))
        m_new[:len_m, :len_m] = matrix
        return m_new

    def _extend_vector(self, vector):
        """
        Extend the dimensionality of vector by one element
        """
        len_v = len(vector)
        v_new = np.zeros(len_v + 1)
        v_new[:len_v] = vector
        return v_new

    def _extend_ktildainv(self, k_tilda_inv, a, delta_new):
        """
        # grows n x n -> n+1 x n+1 where n is dict size
        :returns: inverse of the Gram matrix using the previous Gram matrix and partition inverse theorem
        """
        len_d = len(self.params['_dictionary'])
        k_tilda_inv_new = np.zeros((len_d + 1, len_d + 1))
        k_tilda_inv_new[:len_d, :len_d] = k_tilda_inv + np.outer(a, a) / delta_new
        k_tilda_inv_new[:len_d, len_d] = -a / delta_new  # new col
        k_tilda_inv_new[len_d, :len_d] = -a / delta_new  # new row
        k_tilda_inv_new[-1][-1] = 1 / delta_new  # new corner
        return k_tilda_inv_new

    def _extend_new(self, delta_new, state, action, kernel, _a_new, k_tilda_prev, k_tilda_new, delta_k_tilda_new):
        """
        Add new state and action to the dictionary and extend sufficient statistics matrices and vectors for one dimension
        and reestimates all parameters apart form the ones involving the reward
        """
        _K_tilda_inv_new = self._extend_ktildainv(self.params['_K_tilda_inv'], _a_new, delta_new)
        _a_new = np.zeros(len(self.params['_dictionary']) + 1)
        _a_new[-1] = 1.0
        _h_tilda_new = self._extend_vector(self.params['_a'])
        _h_tilda_new[-1] = - self._gamma

        if self._prior is not None:
            kernel_value = kernel.prior_kernel(state, state)
        else:
            kernel_value = kernel.belief_kernel(state, state)

        delta_k_new = np.dot(self.params['_a'], (k_tilda_prev - 2.0 * self._gamma * k_tilda_new)) \
            + (self._gamma ** 2) * kernel_value * kernel.action_kernel(action, action)

        part1 = np.dot(self.params['_C_tilda'], delta_k_tilda_new)
        part2 = np.zeros(len(self.params['_dictionary'])) \
            if self.initial else (((self._gamma * (self._sigma ** 2)) * self.params['_c_tilda']) / self.params['_s'])

        _c_tilda_new = self._extend_vector(_h_tilda_new[:-1] - part1 + part2)
        _c_tilda_new[-1] = _h_tilda_new[-1]

        spart1 = (1.0 + (self._gamma ** 2)) * (self._sigma ** 2)
        spart2 = np.dot(delta_k_tilda_new, np.dot(self.params['_C_tilda'], delta_k_tilda_new))
        spart3 = 0.0 if self.initial else ((2*np.dot(self.params['_c_tilda'], delta_k_tilda_new)
                                            - self._gamma*(self._sigma**2))
                                           * (self._gamma * (self._sigma ** 2)) / self.params['_s'])

        _s_new = spart1 + delta_k_new - spart2 + spart3
        _alpha_tilda_new = self._extend_vector(self.params['_alpha_tilda'])
        _C_tilda_new = self._extend_matrix(self.params['_C_tilda'])

        self.params['_s'] = _s_new
        self.params['_alpha_tilda'] = _alpha_tilda_new
        self.params['_c_tilda'] = _c_tilda_new
        self.params['_C_tilda'] = _C_tilda_new
        self.params['_K_tilda_inv'] = _K_tilda_inv_new
        self.params['_a'] = _a_new
        self.params['_dictionary'].append([state, action])

    def _no_extend(self, _a_new, delta_k_tilda_new):
        """
        Resestimates sufficient statistics without extending the dictionary
        """
        _h_tilda_new = self.params['_a'] - self._gamma * _a_new

        part1 = np.zeros(len(self.params['_dictionary'])) \
            if self.initial else (self.params['_c_tilda'] * (self._gamma * (self._sigma ** 2)) / self.params['_s'])
        part2 = np.dot(self.params['_C_tilda'], delta_k_tilda_new)
        _c_tilda_new = part1 + _h_tilda_new - part2

        spart1 = (1.0 + (0.0 if self.terminal else (self._gamma ** 2))) * (self._sigma ** 2)
        spart2 = np.dot(delta_k_tilda_new, (_c_tilda_new + (np.zeros(len(self.params['_dictionary']))
                                                            if self.initial else (
                self.params['_c_tilda'] * self._gamma * (self._sigma ** 2) / self.params['_s']))))
        spart3 = (0 if self.initial else ((self._gamma ** 2) * (self._sigma ** 4) / self.params['_s']))

        _s_new = spart1 + spart2 - spart3
        self.params['_c_tilda'] = _c_tilda_new
        self.params['_s'] = _s_new
        self.params['_a'] = _a_new


class Kernel(object):
    """
    The Kernel class defining the kernel for the GPSARSA algorithm.

    The kernel is usually divided into a belief part where a dot product or an RBF-kernel is used.
    The action kernel is either the delta function or a handcrafted or distributed kernel.
    """
    def __init__(self, kernel_type, theta, der=None, action_kernel_type='delta', action_names=None):
        self.kernel_type = kernel_type
        self.action_kernel_type = action_kernel_type

    def action_kernel(self, na, a):
        """
        Kroneker delta on actions
        """
        if self.action_kernel_type == 'delta':
            return 1.0 if na.act == a.act else 0.0

    def prior_kernel(self, ns, s):
        core = self.belief_kernel(ns, s)
        ns_kernel = self.belief_kernel(ns, ns)
        s_kernel = self.belief_kernel(s, s)

        return core / math.sqrt(ns_kernel * s_kernel)

    def belief_kernel(self, ns, s):
        if ns.is_abstract != s.is_abstract:
            exit('Error::Cant compare abstracted and real beliefs - check your config settings!')
        ker = 0.0
        # Calculate actual kernel
        if self.kernel_type == 'polysort':
            if len(ns.belief_state_vec) == len(s.belief_state_vec):
                ker = np.dot(s.belief_state_vec, ns.belief_state_vec)
            else:
                print 'Warning:: ns: {}'.format(len(ns.belief_state_vec)), 's: {}'.format(len(s.belief_state_vec))
                ker = 0
        return ker


class GPState(State):
    """
    Definition of state representation needed for GP-SARSA algorithm
    Main requirement for the ability to compute kernel function over two states
    """
    def __init__(self, belief_state, replace=None):
        super(GPState, self).__init__(belief_state)
        self.is_abstract = True if replace is not None and len(replace) else False
        self.b_state = {}
        self.belief_state_vec = None

        if belief_state is not None:
            self.b_state = PolicyUtils.extract_simple_belief(belief_state, replace)
        self.belief_state_vec = PolicyUtils.belief2vec(self.b_state)

    def to_string(self):
        """
        String representation of the belief
        """
        res = ""
        if len(self.b_state) > 0:
            res += str(len(self.b_state)) + " "
            for slot in self.b_state:
                res += slot + " "
                for elem in self.b_state[slot]:
                    res += str(elem) + " "
        return res

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()


class GPAction(Action):
    """
    Definition of summary action used for GP-SARSA.
    """

    def __init__(self, action, num_actions, replace=None):
        super(GPAction, self).__init__(action)

        # self.act = action
        self.num_actions = num_actions
        # record whether this state has been abstracted -
        self.is_abstract = True if replace is not None and len(replace) else False

        # append to the action the sampled Q value from when we chose it --> for access in batch calculations later
        self.actions_sampled_q_value = 0
        self.likelihood_choose_action = 0

        if replace is not None and len(replace) > 0:
            self.act = self.replace_action(action, replace)

    def replace_action(self, action, replace):
        """
        Used for making abstraction of an action
        """
        if "_" in action:
            slot = action.split("_")[1]
            if slot in replace:
                replacement = replace[slot]
                return action.replace(slot, replacement)  # .replace() is a str operation
        return action

    def __eq__(self, a):
        """
        Action are the same if their strings match
        :rtype : bool
        """
        if self.num_actions != a.num_actions:
            return False
        if self.act != a.act:
            return False
        return True

    def __ne__(self, a):
        return not self.__eq__(a)

    def to_string(self):
        """
        Prints action
        """
        return self.act

    def __str__(self):
        return self.act

    def __repr__(self):
        return self.to_string()


class TerminalGPState(GPState, TerminalState):
    """
    Basic object to explicitly denote the terminal state. Always transition into this state at dialogues completion.
    """
    def __init__(self):
        super(TerminalGPState, self).__init__(None)


class TerminalGPAction(GPAction, TerminalAction):
    """
    Class representing the action object recorded in the (b,a) pair along with the final reward.
    """
    def __init__(self):
        super(TerminalGPAction, self).__init__(None, None)
        self.actions_sampled_q_value = None
        self.likelihood_choose_action = None
        self.act = 'TerminalGPAction'
        self.is_abstract = None
        self.num_actions = None
