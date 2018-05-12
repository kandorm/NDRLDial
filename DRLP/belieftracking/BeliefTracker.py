import pprint
import math
from collections import defaultdict
from DRLP.ontology import Ontology
from DRLP.utils import dact
from DRLP.belieftracking import BeliefTrackingUtils


class BeliefTracker(object):

    def __init__(self):
        self.prev_belief = None
        self.turn = 0

    def restart(self):
        """
        Reset some private members
        """
        self.prev_belief = None
        self.turn = 0

    def _update_belief(self, turn):
        """
        Update the belief given the current turn info
        """
        pass

    def update_belief_state(self, last_act, obs):
        """
        Does the actual belief tracking via tracker.addTurn

        :param last_act: last system dialgoue act
        :type last_act: string

        :param obs: current observation
        :type obs: list

        :return: dict -- previous belief state
        """
        cur_turn = self._convertHypToTurn(last_act, obs)

        if self.turn == 0:
            self.prev_belief = self._init_belief()

        self.prev_belief = self._update_belief(cur_turn)
        self.turn += 1

        return self.prev_belief

    def str(self):
        return pprint.pformat(self.prev_belief)

    ##################################################
    # private methods
    ##################################################

    def _convertHypToTurn(self, last_act, obs):
        """
        Convert hypotheses to turn

        :param last_act: last system dialogue act
        :type last_act: string

        :param obs: current observation
        :type obs: list

        :return: dict -- turn dict
        """
        cur_turn = {'turn-index': self.turn}

        # Last system action
        sys_last_act = []
        if self.turn > 0:
            sys_last_act = dact.parse_act(last_act, user=False)
            sys_last_act = BeliefTrackingUtils.transform_act(sys_last_act, {}, Ontology.global_ontology.get_ontology(),
                                                            user=False)
        cur_turn['output'] = {'dialog-acts': sys_last_act}

        # User act hyps
        accumulated = defaultdict(float)
        for (hyp, prob) in obs:
            hyp = dact.parse_act(hyp)
            hyp = BeliefTrackingUtils.transform_act(hyp, {}, Ontology.global_ontology.get_ontology())
            hyp = dact.infer_slots_for_act(hyp)

            prob = min(1.0, prob)
            if prob < 0:
                prob = math.exp(prob)
            accumulated = BeliefTrackingUtils.addprob(accumulated, hyp, prob)
        sluhyps = BeliefTrackingUtils.normaliseandsort(accumulated)

        cur_turn['input'] = {'live': {'asr-hyps': [], 'slu-hyps': sluhyps}}
        return cur_turn

    def _init_belief(self):
        """
        Simply constructs the belief state data structure at turn 0

        :return: dict -- initiliased belief state
        """
        belief = {}
        for slot in Ontology.global_ontology.get_informable_slots():
            inform_slot_values = Ontology.global_ontology.get_informable_slot_values(slot)
            if slot not in Ontology.global_ontology.get_system_requestable_slots():
                belief[slot] = dict.fromkeys(inform_slot_values, 0.0)
            else:
                belief[slot] = dict.fromkeys(inform_slot_values + ['dontcare'], 0.0)
            belief[slot]['none'] = 1.0
        belief['request'] = dict.fromkeys(Ontology.global_ontology.get_requestable_slots(), 0.0)
        return belief
