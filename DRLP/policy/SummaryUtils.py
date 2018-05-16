import copy
from scipy.stats import entropy
from DRLP.ontology import Ontology
from DRLP.utils import Settings

REQUESTING_THRESHOLD = 0.5


"""
Methods for belief state.
"""


def getTopBelief(slot_belief):
    """
    Return slot value with the largest belief
    :param slot_belief: dict of value-prob pairs for slot distribution
    :return: top_value (str), top_belief (float)
    """
    top_value = max(slot_belief, key=slot_belief.get)
    return top_value, slot_belief[top_value]


def getTopBeliefs(belief_state, threshold='auto'):
    """
    Get slot values with belief larger than threshold.
    :param belief_state: dict representing the full belief state
    :param threshold: threshold on slot value probabilities. Default value is 'auto', only allowable string
    :return: (dict) as {slot: (topvalue, topbelief), ...}
    """
    top_beliefs = {}
    for slot in Ontology.global_ontology.get_system_requestable_slots():
        if threshold == 'auto':
            num_values = Ontology.global_ontology.get_length_informable_slot(slot)
            thres = 1. / (float(num_values) - 0.1)
        else:
            thres = threshold
        top_value, top_belief = getTopBelief(belief_state[slot])
        if top_value != 'none' and top_belief > thres:
            top_beliefs[slot] = (top_value, top_belief)
    return top_beliefs


def getTopBeliefsExcludingNone(slot_belief):
    """
    Get the ordered list of (value,belief) in slot exclude none.
    :param slot_belief: dict of value-prob pairs for slot distribution
    :return: (list) of ordered value-beliefs, (bool) telling if the top value is **NONE**
    """
    slot_belief_copy = copy.deepcopy(slot_belief)
    top_hyps = []
    is_top_none = False
    while len(slot_belief_copy) > 0:
        top_value, top_belief = getTopBelief(slot_belief_copy)
        if len(top_hyps) == 0 and top_value == 'none':
            is_top_none = True
        if top_value != 'none':
            top_hyps.append((top_value, top_belief))
        del slot_belief_copy[top_value]
    return top_hyps, is_top_none


def arraySlotSummary(belief):
    """
    Gets the summary vector for goal slots, including the top probabilities, entropy, etc.
    :param belief: dict representing the full belief state
    :return: (dict) of slot goal summaries
    """
    summary = {}
    slots = Ontology.global_ontology.get_system_requestable_slots()
    for slot in slots:
        summary[slot] = {}
        slot_belief = belief[slot]
        summary[slot]['TOPHYPS'], summary[slot]['ISTOPNONE'] = getTopBeliefsExcludingNone(belief[slot])
        belief_dist = slot_belief.values()
        summary[slot]['ENTROPY'] = entropy(belief_dist)
        summary[slot]['ISREQUESTTOP'] = belief['request'][slot] > REQUESTING_THRESHOLD
    return summary


def getConstraints(accepted_values):
    constraints = {}
    for slot in accepted_values:
        constraints[slot] = accepted_values[slot][0]
    return constraints


def getRequestedSlots(belief_state):
    """
    Iterate get the list of mentioned requested slots

    :param belief_state: dict representing the full belief state
    :return: (list) of slot names with prob retrieved from belief > REQUESTING_THRESHOLD (an internal global)
    """
    requested_slots = []
    for slot in belief_state['request']:
        request_prob = belief_state['request'][slot]
        if request_prob > REQUESTING_THRESHOLD:
            requested_slots.append(slot)
    return requested_slots


"""
Methods for inform related actions.
"""


def getInformByConstraints(constraints, entities):
    """
    Looks for a database match with constraints and converts this entity into a dialogue act.
    The dialogue act for summary action 'inform'.

    :param constraints: dict of slot:values whose beliefs are above 'none'
    :param entities: search results from knowledge base (match to the constraints, if no constraints: random 10)
    :return: string representing the inform dialogue act
    """
    if len(entities) == 0:
        return _getInformNoneVenue(constraints)
    else:
        ret_ent = Settings.random.choice(entities)
        return _getInformEntity(constraints, ret_ent)


def getInformRequestedSlots(requested_slots, name, constraints, entities):
    """
    Informs about the requested slots from the last informed venue of form the venue informed by name
    The dialogue act for summary action 'inform_byname'.

    :param requested_slots: list of requested slots
    :param name: name of the last informed venue
    :param entities: search results from knowledge base (match to the constraints, if no constraints: random 10)
    :return: string representing the inform dialogue act
    """
    if name in ['none', None]:
        # Return a random venue
        if len(entities) > 0:
            ent = Settings.random.choice(entities)
            return _getInformRequestedSlotsForEntity(requested_slots, ent)
        else:
            return _getInformNoneVenue(constraints)

    else:
        result = [ent for ent in entities if ent['name'] == name]
        if len(result) > 0:
            ent = Settings.random.choice(result)
            return _getInformRequestedSlotsForEntity(requested_slots, ent)
        else:
            if len(entities) > 0:
                ent = Settings.random.choice(entities)
                return _getInformRequestedSlotsForEntity(requested_slots, ent)
            else:
                return _getInformNoneVenue(constraints)


def getInformAlternativeEntities(accepted_values, recommended_list, entities):
    constraints = getConstraints(accepted_values)
    if len(entities) == 0:
        return _getInformNoneVenue(constraints)
    else:
        Settings.random.shuffle(entities)
        for ent in entities:
            name = ent['name']
            if name not in recommended_list:
                return _getInformEntity(accepted_values, ent)

        return _getInformNoMoreVenues(accepted_values, entities)


def _convert_feats_to_str(feats):
    result = []
    for slot in feats:
        value = feats[slot]
        if value is not None and value.lower() != 'not available' and value != '':
            result.append('{}="{}"'.format(slot, value))
    return ','.join(result)


def _getInformEntity(accepted_values, ent):
    """
    Converts a database entity into a dialogue act

    :param accepted_values: dict of slot-values whose beliefs are above **NONE**
    :param ent: database entity to be converted to dialogue act
    :return: string representing the inform dialogue act
    """
    feats = {'name': ent['name']}
    num_feats = len(accepted_values)
    acceptance_keys = accepted_values.keys()

    max_num_feats = 5
    if Settings.config.has_option("summaryacts", "maxinformslots"):
        max_num_feats = int(Settings.config.get('summaryacts', 'maxinformslots'))

    if num_feats > max_num_feats:
        Settings.random.shuffle(acceptance_keys)
        acceptance_keys = acceptance_keys[:max_num_feats]

    for slot in acceptance_keys:
        if slot != 'name':
            value = accepted_values[slot]
            if value == 'dontcare' and slot in ent and ent[slot] != "not available":
                feats[slot] = ent[slot]
            else:
                if slot in ent:
                    feats[slot] = ent[slot]
                else:
                    print 'Slot {} is not found in data for entity {}'.format(slot, ent['name'])

    return 'inform({})'.format(_convert_feats_to_str(feats))


def _getInformNoneVenue(constraints):
    """
    creates inform(name=none,...) act

    :param constraints: dict of accepted slot-values
    :return: (str) inform(name=none,...) act
    """
    feats = {}
    for slot in constraints:
        if slot != 'name':
            if constraints[slot] != 'dontcare':
                feats[slot] = constraints[slot]
    if not feats:
        return 'inform(name=none)'
    else:
        return 'inform(name=none, {})'.format(_convert_feats_to_str(feats))


def _getInformRequestedSlotsForEntity(requested_slots, ent):
    """
    Converts the list of requested slots and the entity into a inform_requested dialogue act

    :param requested_slots: list of requested slots (obtained in getRequestedSlots())
    :param ent: dictionary with information about a database entity
    :return: string representing the dialogue act
    """

    slot_value_pair = ['name="{}"'.format(ent['name'])]
    if len(requested_slots) == 0:
        if 'type' in ent:
            slot_value_pair.append('type="{}"'.format(ent['type']))
        else:
            # type is not part of some ontologies. in this case just add a random slot-value
            slots = Ontology.global_ontology.get_requestable_slots()
            if 'name' in slots:
                slots.remove('name')
            slot = slots[Settings.random.randint(len(slots))]
            try:
                slot_value_pair.append('{}="{}"'.format(slot, ent[slot]))
            except KeyError:
                print ent
    else:
        max_num_feats = 5
        if Settings.config.has_option("summaryacts", "maxinformslots"):
            max_num_feats = int(Settings.config.get('summaryacts', 'maxinformslots'))

        if len(requested_slots) > max_num_feats:
            Settings.random.shuffle(requested_slots)
            requested_slots = requested_slots[:max_num_feats]

        for slot in requested_slots:
            if slot != 'name':
                if slot in ent:
                    slot_value_pair.append('{}="{}"'.format(slot, ent[slot]))
                else:
                    slot_value_pair.append('{}=none'.format(slot))

    return 'inform({})'.format(','.join(slot_value_pair))


def _getInformNoMoreVenues(accepted_values, entities):
    """
    returns inform(name=none, other than x and y, with constraints w and z) act

    :param accepted_values: dict of slot-value-beliefs whose beliefs are above **NONE**
    :param entities: list of database entity dicts
    :return: (str) inform() action
    """

    max_num_feats = 5
    if Settings.config.has_option("summaryacts", "maxinformslots"):
        max_num_feats = int(Settings.config.get('summaryacts', 'maxinformslots'))

    feats = {}
    for slot in accepted_values:
        value = accepted_values[slot][0]
        if slot != 'name' or value != 'dontcare':
            feats[slot] = value

    if len(feats) > max_num_feats:
        feats_keys = feats.keys()
        truncated_feats = {}
        Settings.random.shuffle(feats_keys)
        for key in feats_keys[:max_num_feats]:
            truncated_feats[key] = feats[key]
        feats = truncated_feats

    prohibited_list = ''
    for ent in entities:
        prohibited_list += 'name!="{}",'.format(ent['name'])

    return 'inform(name=none,{}{})'.format(prohibited_list, _convert_feats_to_str(feats))
