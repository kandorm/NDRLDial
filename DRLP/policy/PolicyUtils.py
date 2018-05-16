import numpy as np
import scipy.signal
import copy
from DRLP.ontology import Ontology


def flatten_belief(belief_state):
    b_state = {}
    if belief_state is not None:
        b_state = extract_simple_belief(belief_state)
    belief_state_vec = self._slow_to_fast_belief(b_state)

    return belief_state_vec


def belief2vec(belief_dic):
    """
    Converts dictionary format to numpy vector format.
    Ordered by the key of belief_dic.
    """
    values = np.array([])
    for slot in sorted(belief_dic.keys()):
        values = np.concatenate((values, np.array(belief_dic[slot])))
    return values


def extract_simple_belief(belief_state, replace=None):
    """
    From the belief state extracts requested slots, name, goal for each slot.
    Sets self._bstate
    """
    _bstate = {}
    for elem in belief_state.keys():
        if elem == 'request':
            for slot in belief_state[elem]:
                cur_slot = slot
                if replace is not None and len(replace) > 0:
                    cur_slot = replace[cur_slot]
                _bstate['hist_' + cur_slot] = _extract_single_value(belief_state[elem][slot])
        elif elem == 'user_intent':
            for slot in belief_state[elem]:
                _bstate['intent_' + slot] = _extract_single_value(belief_state[elem][slot])

        else:
            if elem == 'name':
                pass
            else:
                cur_slot = elem
                if replace is not None and len(replace) > 0:
                    cur_slot = replace[elem]
                _bstate['goal_' + cur_slot] = _extract_belief_with_other(belief_state[elem])

                num_additional_slot = 2
                if len(_bstate['goal_' + cur_slot]) != \
                        Ontology.global_ontology.get_length_informable_slot(elem) + num_additional_slot:
                    print _bstate['goal_' + cur_slot]
                    exit('Different number of values for slot' + cur_slot + ' ' +
                         str(len(_bstate['goal_' + cur_slot])) + ' in ontology ' +
                         str(Ontology.global_ontology.get_length_informable_slot(elem) + num_additional_slot))
    return _bstate


def _extract_single_value(val):
    """
    For a probability p returns a list  [p,1-p]
    """
    return [val, 1-val]


def _extract_belief_with_other(val_and_belief, sort=True):
    """
    Copies a belief vector, computes the remaining belief, appends it and return its sorted value.
    The first one is the belief of 'none' and the others(include 'dontcare') arranged in descending order.
    :return: the sorted belief state value vector
    """
    v_b = copy.deepcopy(val_and_belief)
    res = []

    if 'none' not in v_b:
        res.append(1.0 - sum(v_b.values()))     # append the none probability
    else:
        res.append(v_b['none'])
        del v_b['none']

    # ensure that all goal slots have dontcare entry for GP belief representation
    if 'dontcare' not in v_b:
        v_b['dontcare'] = 0.0

    if sort:
        res.extend(sorted(v_b.values(), reverse=True))
    else:
        res.extend(v_b.values())

    return res


def get_state_dim(use_alter=False):
    dim = 0
    slots = Ontology.global_ontology.get_informable_slots()
    for slot in slots:
        dim += Ontology.global_ontology.get_length_informable_slot(slot)
        dim += 2    # for 'none' and 'dontcare'

    dim += Ontology.global_ontology.get_length_requestable_slots()

    if use_alter:
        dim += 1

    return dim


def softmax(x, t=1.0):
    """
    Calculate the softmax of a list of numbers x.
    """

    x = np.array(x) / t
    e_x = np.exp(x - np.max(x))
    dist = e_x / e_x.sum()
    return dist


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
