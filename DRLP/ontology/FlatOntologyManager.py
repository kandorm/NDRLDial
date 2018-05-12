import json
import codecs
import copy
from DRLP.utils import Settings


class FlatOntologyManager(object):

    def __init__(self):
        ontology_file = 'DRLP/ontology/ontologies/CamRestaurants-rules.json'
        if Settings.config.has_option('file', 'ontology'):
            ontology_file = Settings.config.get('file', 'ontology')
        try:
            self.ontology = json.load(codecs.open(ontology_file, "r", "utf-8"))
        except Exception as e:
            print e

    def get_ontology(self):
        return self.ontology

    def get_type(self):
        return self.ontology['type']

    def get_requestable_slots(self):
        return copy.copy(self.ontology['requestable'])

    def get_length_requestable_slots(self):
        return len(self.ontology['requestable'])

    def get_system_requestable_slots(self):
        return copy.copy(self.ontology['system_requestable'])

    def get_length_system_requestable_slots(self):
        return len(self.ontology['system_requestable'])

    def get_user_requestable_slots(self):
        all_req = self.ontology['requestable']
        sys_req = self.ontology['system_requestable']
        request_slots = list(set(all_req) - set(sys_req))
        return request_slots

    def is_user_requestable(self, slot):
        logic1 = slot in self.ontology['requestable']
        logic2 = slot not in self.ontology['system_requestable']
        if logic1 and logic2:
            return True
        else:
            return False

    def is_system_requestable(self, slot):
        if slot in self.ontology['system_requestable']:
            return True
        else:
            return False

    def get_informable_slots(self):
        return self.ontology['informable'].keys()

    def get_informable_slots_and_values(self):
        return self.ontology['informable']

    def get_informable_slot_values(self, slot):
        try:
            return copy.copy(self.ontology["informable"][slot])
        except KeyError:
            print KeyError('FlatOntologyManager.get_informable_slot_values')

    def get_length_informable_slot(self, slot):
        return len(self.ontology['informable'][slot])

    def get_random_value_for_slot(self, slot, no_dontcare=False, no_these=None):
        """
        :param slot: None
        :type slot: str
        :param no_dontcare: None
        :type no_dontcare: bool
        :param no_these: None
        :type no_these: list
        """
        if slot not in self.ontology["informable"]:
            return None

        if no_these is None:
            no_these = []

        candidate = copy.deepcopy(self.ontology['informable'][slot])
        if len(candidate) == 0:
            print "candidates for slot {} should not be empty".format(slot)
        if not no_dontcare:
            candidate += ['dontcare']
        candidate = list(set(candidate) - set(no_these))
        if len(candidate) == 0:
            return 'dontcare'
        return Settings.random.choice(candidate)
