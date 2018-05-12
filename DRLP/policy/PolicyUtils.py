import numpy as np
import scipy.signal
from DRLP.ontology import Ontology


def flatten_belief(belief_state):
    policy_features = Ontology.global_ontology.get_system_requestable_slots() + ['name', 'request']
    flat_belief = []
    for feat in policy_features:
        if feat == 'request':
            add_feature = [belief_state['request'][slot] for slot in Ontology.global_ontology.get_requestable_slots()]
        else:
            add_feature = [belief_state[feat][value] for value in
                           Ontology.global_ontology.get_informable_slot_values(feat)]
            try:
                add_feature.append(belief_state[feat]['dontcare'])
            except KeyError:
                add_feature.append(0.)  # for dontcare
            try:
                add_feature.append(belief_state[feat]['none'])
            except KeyError:
                add_feature.append(0.)  # for NONE

        flat_belief += add_feature

    return flat_belief


def get_state_dim():
    dim = 0
    slots = Ontology.global_ontology.get_informable_slots()
    for slot in slots:
        dim += Ontology.global_ontology.get_length_informable_slot(slot)
        dim += 2    # for 'none' and 'dontcare'

    dim += Ontology.global_ontology.get_length_requestable_slots()

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
