import copy
from collections import defaultdict
from DRLPolicy.ontology import Ontology
from DRLPolicy.belieftracking.BeliefTracker import BeliefTracker

"""
Belief State Representation
{
    u'informable_slot1': {
                            'none': 1.0,
                            'dontcare': 0.0,
                            u'value1': 0.0,
                            u'value2': 0.0,
                            ......
                         },
    u'informable_slot2': {
                            'none': 1.0,
                            'dontcare': 0.0,
                            u'value1': 0.0,
                            u'value2': 0.0,
                            ......
                         },
    u'name': {
                'none': 1.0,
                u'value1': 0.0,
                u'value2': 0.0,
                ......
             }
    'request': {
                    u'infromable_slot1': 0.0,
                    u'informable_slot2': 0.0,
                    ......
                    u'requestable_slot1': 0.0,
                    u'requestable_slot2': 0.0,
                    u'name': 0.0
                }
}
"""


def labels(user_act, mact):
    """
    Convert inputs to compatible info to belief state

    :param user_act: user's dialogue acts
    :type user_act: dict

    :param mact: machine's dialogue acts
    :type mact: dict

    :return: informed_goals, denied_goals, requested_slots
    """
    # get context for "this" in inform(=dontcare)
    # get context for affirm and negate
    this_slot = None

    confirm_slots = {"explicit": [], "implicit": []}
    for act in mact:
        if act["act"] == "request":
            this_slot = act["slots"][0][1]
        elif act["act"] == "select":
            this_slot = act["slots"][0][0]
        elif act["act"] == "impl-conf":
            confirm_slots["implicit"] += act["slots"]
        elif act["act"] == "expl-conf":
            confirm_slots["explicit"] += act["slots"]
            this_slot = act["slots"][0][0]

    # goal_labels
    informed_goals = {}
    denied_goals = defaultdict(list)
    for act in user_act:
        act_slots = act["slots"]
        slot = None
        value = None
        if len(act_slots) > 0:
            assert len(act_slots) == 1

            if act_slots[0][0] == "this":
                slot = this_slot
            else:
                slot = act_slots[0][0]
            value = act_slots[0][1]

        if act["act"] == "inform" and slot is not None:
            informed_goals[slot] = value

        elif act["act"] == "deny" and slot is not None:
            denied_goals[slot].append(value)

        elif act["act"] == "negate":
            slot_values = confirm_slots["implicit"] + confirm_slots["explicit"]
            if len(slot_values) > 1:
                # print "Warning: negating multiple slots- it's not clear what to do."
                pass
            else:
                for slot, value in slot_values:
                    denied_goals[slot].append(value)

        elif act["act"] == "affirm":
            slot_values = confirm_slots["explicit"]
            if len(slot_values) > 1:
                # print "Warning: affirming multiple slots- it's not clear what to do."
                pass
            else:
                for slot, value in confirm_slots["explicit"]:
                    informed_goals[slot] = value

    # requested slots
    requested_slots = []
    for act in user_act:
        if act["act"] == "request":
            for _, requested_slot in act["slots"]:
                requested_slots.append(requested_slot)
        if act["act"] == "confirm":  # added by dk449
            for requested_slot, _ in act["slots"]:
                requested_slots.append(requested_slot)

    return informed_goals, denied_goals, requested_slots


def Uacts(turn):
    """
    Convert turn info to hypotheses
    :param turn:
    :type turn: dict

    :return: list -- converted hypotheses
    """
    # return merged slu-hyps, replacing "this" with the correct slot
    mact = []
    if "dialog-acts" in turn["output"]:
        mact = turn["output"]["dialog-acts"]
    this_slot = None
    for act in mact:
        if act["act"] == "request":
            this_slot = act["slots"][0][1]
    this_output = []
    for slu_hyp in turn['input']["live"]['slu-hyps']:
        score = slu_hyp['score']
        this_slu_hyp = slu_hyp['slu-hyp']
        these_hyps = []
        for hyp in this_slu_hyp:
            for i in range(len(hyp["slots"])):
                slot, _ = hyp["slots"][i]
                if slot == "this":
                    hyp["slots"][i][0] = this_slot
            these_hyps.append(hyp)
        this_output.append((score, these_hyps))
    this_output.sort(key=lambda x: x[0], reverse=True)
    return this_output


class RuleBasedTracker(BeliefTracker):

    def __init__(self):
        super(RuleBasedTracker, self).__init__()

    def _update_belief(self, turn):
        """
        Update belief state

        :param turn:
        :type turn: dict

        :return: None
        """
        track = self._add_turn(turn)
        return self._tobelief(self.prev_belief, track)

    def _add_turn(self, turn):
        """
        Add turn info

        :param turn:
        :type turn: dict

        :return: None
        """
        pass

    def _tobelief(self, prev_belief, track):
        """
        Add up previous belief and current tracking result to current belief

        :param prev_belief:
        :type prev_belief: dict

        :param track:
        :type track: dict

        :return: dict -- belief state
        """
        belief = {}
        for slot in Ontology.global_ontology.get_informable_slots():
            if slot in track['goal-labels']:
                infom_slot_values = Ontology.global_ontology.get_informable_slot_values(slot)
                belief[slot] = dict.fromkeys(infom_slot_values + ['dontcare'], 0.0)
                for v in track['goal-labels'][slot]:
                    belief[slot][v] = track['goal-labels'][slot][v]
                belief[slot]['none'] = 1.0 - sum(belief[slot].values())
            else:
                belief[slot] = prev_belief[slot]
        belief['request'] = dict.fromkeys(Ontology.global_ontology.get_requestable_slots(), 0.0)
        for v in track['requested-slots']:
            belief['request'][v] = track['requested-slots'][v]
        return belief


class FocusTracker(RuleBasedTracker):
    """
    It accumulates evidence and has a simple model of how the state changes throughout the dialogue.
    Only track goals but not requested slots and method.
    """

    def __init__(self):
        super(FocusTracker, self).__init__()
        self.hyps = {"goal-labels": {}, "requested-slots": {}}

    def _add_turn(self, turn):
        """
        Add turn info

        :param turn:
        :type turn: dict

        :return: None
        """
        hyps = copy.deepcopy(self.hyps)
        if "dialog-acts" in turn["output"]:
            mact = turn["output"]["dialog-acts"]
        else:
            mact = []
        slu_hyps = Uacts(turn)

        this_u = defaultdict(lambda: defaultdict(float))
        requested_slot_stats = defaultdict(float)
        for score, uact in slu_hyps:
            informed_goals, denied_goals, requested = labels(uact, mact)
            for slot in requested:
                requested_slot_stats[slot] += score
            # goal_labels
            for slot in informed_goals:
                this_u[slot][informed_goals[slot]] += score

        for slot in set(this_u.keys() + hyps["goal-labels"].keys()):
            q = max(0.0, 1.0 - sum(
                [this_u[slot][value] for value in this_u[slot]]))  # clipping at zero because rounding errors
            if slot not in hyps["goal-labels"]:
                hyps["goal-labels"][slot] = {}

            for value in hyps["goal-labels"][slot]:
                hyps["goal-labels"][slot][value] *= q
            prev_values = hyps["goal-labels"][slot].keys()
            for value in this_u[slot]:
                if value in prev_values:
                    hyps["goal-labels"][slot][value] += this_u[slot][value]
                else:
                    hyps["goal-labels"][slot][value] = this_u[slot][value]

            hyps["goal-labels"][slot] = normalise_dict(hyps["goal-labels"][slot])

        # requested slots
        informed_slots = []
        for act in mact:
            if act["act"] == "inform":
                for slot, value in act["slots"]:
                    informed_slots.append(slot)

        for slot in set(requested_slot_stats.keys() + hyps["requested-slots"].keys()):
            p = requested_slot_stats[slot]
            prev_p = 0.0
            if slot in hyps["requested-slots"]:
                prev_p = hyps["requested-slots"][slot]
            x = 1.0 - float(slot in informed_slots)
            new_p = x * prev_p + p
            hyps["requested-slots"][slot] = clip(new_p)

        self.hyps = hyps
        return self.hyps

    def restart(self):
        """
        Reset the hypotheses
        """
        super(FocusTracker, self).restart()
        self.hyps = {"goal-labels": {}, "requested-slots": {}}


def clip(x):
    if x > 1:
        return 1
    if x < 0:
        return 0
    return x


def normalise_dict(x):
    x_items = x.items()
    total_p = sum([p for k, p in x_items])
    if total_p > 1.0:
        x_items = [(k, p/total_p) for k, p in x_items]
    return dict(x_items)
