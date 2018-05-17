from DRLP.belieftracking.FocusTracker import FocusTracker
from DRLP.policy.PolicyManager import PolicyManager
from DRLP.evaluation.EvaluationManager import EvaluationManager
from DRLP.utils.dact import DiaAct
from DRLP.utils import Settings
from DRLP.ontology import Ontology
from KB import KnowledgeBase


class DialogueAgent(object):

    def __init__(self):
        self.reward = None
        self.task = None
        self.subjective = None

        self.ending_dialogue = False
        self.cur_turn = 0
        self.num_dialogs = 0
        self.recommended_list = []

        self.max_turns = 25
        self.save_frequency = 10

        if Settings.config.has_option('agent', 'savefrequency'):
            self.save_frequency = Settings.config.getint('agent', 'savefrequency')
        if Settings.config.has_option("agent", "maxturns"):
            self.max_turns = Settings.config.getint("agent", "maxturns")

        self.bs_tracker = FocusTracker()
        self.policy_manager = PolicyManager()
        self.evaluation_manager = EvaluationManager()

    def restart(self):
        self.reward = None
        self.task = None
        self.subjective = None

        self.ending_dialogue = False
        self.cur_turn = 0
        self.recommended_list = []

        self.bs_tracker.restart()
        self.policy_manager.restart()
        self.evaluation_manager.restart()

    def retrieve_last_sys_act(self):
        return self.policy_manager.get_last_system_action()

    def start_call(self, simulated_user):
        self.restart()
        self.num_dialogs += 1

        last_sys_act = self.retrieve_last_sys_act()

        # SYSTEM ACT:
        # 1. Belief state tracking
        belief_state = self.bs_tracker.update_belief_state(last_act=last_sys_act, obs=[])

        # 2.Query Knowledge base
        constraints = {}
        slots = Ontology.global_ontology.get_system_requestable_slots()
        for slot in slots:
            constraints[slot] = max(belief_state[slot], key=belief_state[slot].get)
        entities = KnowledgeBase.global_kb.entity_by_features(constraints)

        # 3. Add recommended list into belief state
        belief_state['name'] = self.recommended_list

        # 4. Policy -- Determine system act/response
        sys_act = self.policy_manager.act_on(belief_state, entities)

        # 5. Add system recommend restaurant to the recommended list
        if not isinstance(sys_act, DiaAct):
            sys_act = DiaAct(str(sys_act))

        if sys_act.act == 'inform':
            name = sys_act.get_value('name', negate=False)
            if name not in ['none', None]:
                try:
                    self.recommended_list.remove(name)
                except:
                    pass
                self.recommended_list.append(name)

        # 6. EVALUATION: - record the system action
        self._evaluate_agents_turn(simulated_user, sys_act, belief_state)

        return sys_act

    def continue_call(self, user_act, simulated_user):

        if self._turn_increment_and_check():
            sys_act = DiaAct('bye()')
            return sys_act

        # Make sure there is some asr information:
        if user_act is None:
            sys_act = DiaAct('null()')
            return sys_act

        last_sys_act = self.retrieve_last_sys_act()

        # SYSTEM ACT:
        # 1. Belief state tracking
        belief_state = self.bs_tracker.update_belief_state(last_act=last_sys_act, obs=[(str(user_act), 1.0)])

        # 2.Query Knowledge base
        constraints = {}
        slots = Ontology.global_ontology.get_system_requestable_slots()
        for slot in slots:
            constraints[slot] = max(belief_state[slot], key=belief_state[slot].get)
        entities = KnowledgeBase.global_kb.entity_by_features(constraints)

        # 3. Add recommended list into belief state
        belief_state['name'] = self.recommended_list

        # 4. Policy -- Determine system act/response type: DiaAct
        sys_act = self.policy_manager.act_on(belief_state, entities)

        # 5. Add system recommend restaurant to the recommended list
        if not isinstance(sys_act, DiaAct):
            sys_act = DiaAct(str(sys_act))

        if sys_act.act == 'inform':
            name = sys_act.get_value('name', negate=False)
            if name not in ['none', None]:
                try:
                    self.recommended_list.remove(name)
                except:
                    pass
                self.recommended_list.append(name)

        # 6. EVALUATION: - record the system action
        self._evaluate_agents_turn(simulated_user, sys_act, belief_state)

        return sys_act

    def end_call(self, simulated_user=None, no_training=False):
        """
        Performs end of dialog clean up: policy learning, policy saving and housecleaning. The no_training parameter
        is used in case of an abort of the dialogue where you still want to gracefully end it,
        e.g., if the dialogue server receives a clean request.

        :param no_training: train the policy when ending dialogue
        :type no_training: bool

        :return: None
        """
        final_info = {'task': self.task,
                      'subjectiveSuccess': self.subjective}
        if simulated_user is not None:
            final_info['usermodel'] = simulated_user.um
        final_reward = self.evaluation_manager.get_final_reward(final_info)
        self.policy_manager.finalize_record(final_reward)
        if not no_training:
            self.policy_manager.train(self.evaluation_manager.do_training())
        self._save_policy()
        self.ending_dialogue = False

    def power_down(self):
        self.evaluation_manager.print_summary()
        self._save_policy()

    def _save_policy(self):
        if self.num_dialogs % self.save_frequency == 0:
            self.policy_manager.save_policy()

    def _turn_increment_and_check(self):
        self.cur_turn += 1
        if self.cur_turn > self.max_turns:
            self.ending_dialogue = True
            return True
        return False

    def _evaluate_agents_turn(self, simulated_user=None, sys_act=None, belief_state=None):
        """
        This function needs to record per exchange rewards and pass them to dialogue management.

        :param sys_act: system's dialogue act
        :type sys_act: DiaAct
        :param belief_state: belief state
        :type dict

        :return: None
        """
        if self.cur_turn == 0:
            return
        # 1. Get reward
        # -------------------------------------------------------------------------------------------------------------
        self.reward = None
        turn_info = {'sys_act': str(sys_act),
                     'state': belief_state,
                     'prev_sys_act': self.retrieve_last_sys_act()}
        if simulated_user is not None:
            turn_info['usermodel'] = simulated_user.um

        self.reward = self.evaluation_manager.get_turn_reward(turn_info)

        self.policy_manager.record(self.reward)
