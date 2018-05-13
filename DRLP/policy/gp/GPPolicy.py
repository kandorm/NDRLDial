import json
from DRLP.utils import Settings
from DRLP.policy.Policy import Policy, TerminalState, TerminalAction
from DRLP.policy.gp.GPLib import GPSARSA, GPState, GPAction, Kernel, TerminalGPState, TerminalGPAction


class GPPolicy(Policy):

    def __init__(self, learning, in_policy_file, out_policy_file):
        super(GPPolicy, self).__init__(learning)

        # DEFAULTS:
        self.kernel_type = "polysort"
        self.thetafile = ""
        self.theta = [1.0, 1.0]
        self.action_kernel_type = 'delta'
        self.replace = {}
        self.slot_abstraction_file = 'DRLP/policy/slot_abstractions/CamRestaurants.json'
        self.abstract_slots = False
        self.unabstract_slots = False
        self.use_alter = False

        if Settings.config.has_option('gppolicy', "kernel"):
            self.kernel_type = Settings.config.get('gppolicy', "kernel")
        if Settings.config.has_option('gppolicy', "thetafile"):
            self.thetafile = Settings.config.get('gppolicy', "thetafile")
        if Settings.config.has_option('gppolicy', "actionkerneltype"):
            self.action_kernel_type = Settings.config.get('gppolicy', "actionkerneltype")
        if Settings.config.has_option('gppolicy', "slotabstractionfile"):
            self.slot_abstraction_file = Settings.config.get('gppolicy', "slotabstractionfile")
        if Settings.config.has_option('gppolicy', "abstractslots"):
            self.abstract_slots = Settings.config.getboolean('gppolicy', "abstractslots")
        if not self.abstract_slots:
            if Settings.config.has_option('gppolicy', "unabstractslots"):
                self.unabstract_slots = Settings.config.getboolean('gppolicy', "unabstractslots")

        if Settings.config.has_option('policy', 'use_alter'):
            self.use_alter = Settings.config.getboolean('policy', 'use_alter')

        # Learning algorithm:
        self.learner = GPSARSA(in_policy_file, out_policy_file, learning=self.learning)

        if self.abstract_slots and self.unabstract_slots:   # enforce some logic on your config settings:
            exit('Cant both be true - if abstracting, we keep dictionary abstract. If unabstracting dictionary, '
                 'do so and keep everything in real format. Adjust your config.')

        if self.abstract_slots or self.unabstract_slots:
            self._load_slot_abstraction_mapping()   # loads self.abstraction_mapping
            if self.abstract_slots:
                # if using BCM & we have a GP policy previously with nonempty dictionary not used with BCM before
                self.replace = self.abstraction_mapping['real2abstract']    # will be used throughout
                assert(len(self.replace))
                self._abstract_dictionary()
            if self.unabstract_slots:
                # Un abstract the dictionary if it was used in BCM before and not now:
                self._unabstract_dictionary()

        self._load_theta()
        self.kernel = Kernel(self.kernel_type, self.theta, None, self.action_kernel_type, self.actions.action_names)

    #########################################################
    # overridden methods from Policy
    #########################################################
    def next_action(self, belief_state, entities):
        """
        Selects next action to take based on the current belief and a list of non executable actions
        :param belief_state: the state the policy acts on
        :type belief_state: dict
        :param entities: search results from knowledge base (match to the constraints, if no constraints: random 10)
        :type entities: list
        :param recommended_list: the name list of the restaurant that have been recommended.
        :type recommended_list: list
        :returns: the next system action
        """
        none_executable_actions = self.actions.get_none_executable(belief_state, entities)

        cur_state = GPState(belief_state, replace=self.replace, use_alter=self.use_alter)
        executable = self._create_executable(none_executable_actions)
        if len(executable) < 1:
            exit("No executable actions")

        best_action, actions_sampled_q_value, actions_likelihood = self.learner.policy(
            state=cur_state, kernel=self.kernel, executable=executable)

        summary_act = self._action_string(best_action.act)

        if self.learning:
            best_action.actions_sampled_q_value = actions_sampled_q_value
            best_action.likelihood_choose_action = actions_likelihood

        self.act_to_be_recorded = best_action
        # Finally convert action to MASTER ACTION
        master_act = self.actions.convert(belief_state, summary_act, entities)
        return master_act

    def train(self):
        """
        At the end of learning episode calls LearningStep for accumulated states and actions and rewards
        """
        if self.episode is not None:
            self._process_episode()

    def convert_state_action(self, state, action):
        """
        :param state:
        :type state: dict
        :param action:
        :type action:
        """
        c_state = state
        c_action = action

        if not isinstance(state, GPState):
            if isinstance(state, TerminalState):
                c_state = TerminalGPState()
            else:
                c_state = GPState(state, replace=self.replace, use_alter=self.use_alter)

        if not isinstance(action, GPAction):
            if isinstance(action, TerminalAction):
                c_action = TerminalGPAction()
            else:
                c_action = GPAction(action, self.num_actions, replace=self.replace)

        return c_state, c_action

    def save_policy(self):
        """
        Saves the GP policy.
        """
        if self.learning:
            self.learner.save_policy()

    def _process_episode(self):
        if len(self.episode.state_trace) == 0:
            return
        if not self.learner.learning:
            return
        self.episode.check()  # just checks that traces match up.

        idx = 1
        reward = 0
        while idx < len(self.episode.state_trace) and self.learner.learning:

            prev_GPState = self.episode.state_trace[idx - 1]
            prev_GPAction = self.episode.action_trace[idx - 1]
            cur_GPState = self.episode.state_trace[idx]
            cur_GPAction = self.episode.action_trace[idx]

            self.learner.initial = False
            self.learner.terminal = False

            if idx == 1:
                self.learner.initial = True

            if idx + 1 == len(self.episode.state_trace) or isinstance(self.episode.state_trace[idx], TerminalGPState):
                self.learner.terminal = True
                reward = self.episode.get_weighted_reward()

            self.learner.learning_step(prev_GPState, prev_GPAction, reward, cur_GPState, cur_GPAction, self.kernel)
            idx += 1

            if self.learner.terminal and idx < len(self.episode.state_trace):
                break

    def _load_theta(self):
        """
        Kernel parameters
        """
        if self.thetafile != "":
            f = open(self.thetafile, 'r')
            self.theta = []
            for line in f:
                line = line.strip()
                elements = line.split(" ")
                for elem in elements:
                    self.theta.append(float(elem))
                break
            f.close()

    def _load_slot_abstraction_mapping(self):
        """
        Loads the slot mappings. self.replace does abstraction: request_area --> request_slot0 etc
        """
        with open(self.slot_abstraction_file, 'r') as f:
            self.abstraction_mapping = json.load(f)

    def _unabstract_action(self, action):
        """
        action is a string
        :param action: action
        :type action: str
        """
        if len(action.split("_")) != 2:  # handle not abstracted acts like 'inform' or 'repeat'
            return action
        [prefix, slot] = action.split("_")
        if prefix == 'inform':  # handle not abstracted acts like 'inform_byname' or 'inform_requested'
            return action
        else:  # handle abstracted acts like 'request_slot00' or 'confirm_slot03'
            matching_actions = []
            for abs_slot in self.abstraction_mapping['abstract2real'].keys():
                if abs_slot == slot:
                    match = prefix + '_' + self.abstraction_mapping['abstract2real'][abs_slot]
                    matching_actions.append(match)
            Settings.random.shuffle(matching_actions)
            return Settings.random.choice(matching_actions)

    def _unabstract_dictionary(self):
        for i in range(len(self.learner.params['_dictionary'])):
            # for back compatibility with earlier trained policies
            try:
                _ = self.learner.params['_dictionary'][i][0].is_abstract
            except AttributeError:
                self.learner.params['_dictionary'][i][0].is_abstract = False
                self.learner.params['_dictionary'][i][1].is_abstract = False

            # 0 index is GPState
            if self.learner.params['_dictionary'][i][0].is_abstract:
                for item in self.learner.params['_dictionary'][i][0].b_state:
                    if 'slot' in item:  # covers 'goal_slot01' and 'goal_infoslot00' -- i.e all things abstracted:
                        try:
                            [prefix, slot] = item.split('_')
                            real_name = prefix + '_' + self.abstraction_mapping['abstract2real'][slot]
                            self.learner.params['_dictionary'][i][0].b_state[real_name] = \
                                self.learner.params['_dictionary'][i][0].b_state.pop(item)
                        except KeyError:
                            print '{} - slot not in mapping'.format(item)
                self.learner.params['_dictionary'][i][0].is_abstract = False

            # 1 index is action. should always be the same as state ...
            if self.learner.params['_dictionary'][i][1].is_abstract:
                self.learner.params['_dictionary'][i][1].act = self._unabstract_action(
                    self.learner.params['_dictionary'][i][1].act)
                self.learner.params['_dictionary'][i][1].is_abstract = False

    def _abstract_dictionary(self):
        for i in range(len(self.learner.params['_dictionary'])):
            # for back compatibility with earlier trained policies
            try:
                _ = self.learner.params['_dictionary'][i][0].is_abstract
            except AttributeError:
                self.learner.params['_dictionary'][i][0].is_abstract = False
                self.learner.params['_dictionary'][i][1].is_abstract = False

            # 0 index is state
            if not self.learner.params['_dictionary'][i][0].is_abstract:
                for item in self.learner.params['_dictionary'][i][0].b_state:
                    if '_' in item:  # if not --> not abstract
                        if len(item.split('_')) == 2:  # if not --> not abstract
                            [prefix, slot] = item.split('_')
                            try:
                                abstract_name = prefix + '_' + self.replace[slot]
                                self.learner.params['_dictionary'][i][0].b_state[abstract_name] = \
                                    self.learner.params['_dictionary'][i][0].b_state.pop(item)
                            except KeyError:
                                print '{} - slot not in mapping'.format(item)
                self.learner.params['_dictionary'][i][0].is_abstract = True

            # 1 index is action. abstraction status should always be the same as state ...
            if not self.learner.params['_dictionary'][i][1].is_abstract:
                act = self.learner.params['_dictionary'][i][1].act
                # Use GPAction instance method replace_action() to perform the action abstraction:
                self.learner.params['_dictionary'][i][1].act = self.learner.params['_dictionary'][i][1].replace_action(
                    act, self.replace)
                self.learner.params['_dictionary'][i][1].is_abstract = True

    def _create_executable(self, non_executable_actions):
        """
        Produce a list of executable actions from non executable actions
        :param non_executable_actions:
        :type non_executable_actions:
        """
        executable_actions = []
        for act_i in self.actions.action_names:
            if act_i in non_executable_actions:
                continue
            elif len(self.replace) > 0:  # with abstraction  (ie BCM)
                # check if possibly abstract act act_i is in non_executable_actions
                if '_' in act_i:
                    [prefix, slot] = act_i.split('_')
                    if slot in self.replace.keys():
                        # assumes non_executable_actions is abstract
                        if prefix + '_' + self.replace[slot] not in non_executable_actions:
                            executable_actions.append(GPAction(act_i, self.num_actions, replace=self.replace))
                        else:
                            pass  # dont add in this case
                    else:  # some actions like 'inform_byname' have '_' in name but are not abstracted
                        executable_actions.append(GPAction(act_i, self.num_actions, replace=self.replace))
                else:  # only abstract actions with '_' in them like request_area --> request_slot1 etc
                    executable_actions.append(GPAction(act_i, self.num_actions, replace=self.replace))
            else:  # no abstraction
                executable_actions.append(GPAction(act_i, self.num_actions))  # replace not needed here - no abstraction
        return executable_actions

    def _action_string(self, act):
        """
        Produce a string representation from an action - checking as well that the act coming in is valid
        Should only be called with non abstract action. Use _unabstract_action() otherwise
        :param act:
        :type act:
        """
        if act in self.actions.action_names:
            return act
        exit('Failed to find action %s' % act)
