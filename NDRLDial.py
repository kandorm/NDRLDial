import json
import codecs
from loader.DataReader import DataReader
from utils.dact import DiaAct
from NBT.tracker.net import Tracker
from DRLPolicy.Model import DRLPolicy
from KB.KBManager import KBManager
from SemO.BasicSemOMethod import BasicSemO

from ConfigParser import SafeConfigParser


class NDRLDial(object):

    def __init__(self, config):
        if config is None:
            print "Please specify config file ..."
            return

        parser = SafeConfigParser()
        parser.read(config)

        self.tracker_config = parser.get('config', 'tracker_config')
        self.policy_config = parser.get('config', 'policy_config')

        self.tracker = Tracker(self.tracker_config)
        self.policy_manager = DRLPolicy(self.policy_config)
        # Connect to the knowledge base
        self.kb_manager = KBManager()
        self.generator = BasicSemO()

        self.ontology_file = parser.get('file', 'ontology')
        self.corpus_file = parser.get('file', 'corpus')
        self.reader = DataReader(ontology_file=self.ontology_file, corpus_file=self.corpus_file)

        self.ontology = self.reader.ontology
        self.corpus = self.reader.corpus

        self.name_values = self.ontology['informable']['name'] + ['none']

    def test(self):
        pass

    def reply(self, user_utt, last_sys_act, belief_state):

        response = {'last_sys_act': '',
                    'generated': '',
                    'belief_state': {}}

        if not isinstance(belief_state, dict):
            return response

        sys_req = []
        sys_conf_slot = []
        sys_conf_value = []
        last_sys_act = DiaAct(str(last_sys_act))
        if last_sys_act.act == 'request':
            for item in last_sys_act.items:
                sys_req.append(item.slot)
        elif last_sys_act.act == 'confirm':
            for item in last_sys_act.items:
                sys_conf_slot.append(item.slot)
                sys_conf_value.append(item.val)

        # 1. Belief state tracking (without 'name' slot)
        prediction_dict, distribution_dict, previous_belief_state = \
            self.tracker.track_utterance([(user_utt, 1.0)], sys_req, sys_conf_slot, sys_conf_value, belief_state)

        print prediction_dict

        if 'name' in belief_state:
            distribution_dict['name'] = belief_state['name']
        else:
            distribution_dict['name'] = dict.fromkeys(self.name_values, 0.0)
            distribution_dict['name']['none'] = 1.0

        # 2.Query Knowledge base
        constraints = {}
        slots = self.ontology['system_requestable']
        for slot in slots:
            constraints[slot] = prediction_dict[slot]
        entities = self.kb_manager.entity_by_features(constraints)

        # 3. Policy -- Determine system act/response type: DiaAct
        sys_act = self.policy_manager.act_on(distribution_dict, entities)

        print sys_act

        # 4. Generate natural language
        nl = self.generator.generate(str(sys_act))

        if not isinstance(sys_act, DiaAct):
            sys_act = DiaAct(str(sys_act))

        if sys_act.act == 'inform':
            for item in sys_act.items:
                if item.slot == 'name' and item.op == '=' and item.val not in ['none', 'dontcare']:
                    name_slot = dict.fromkeys(self.name_values, 0.0)
                    name_slot[item.val] = 1.0
                    distribution_dict['name'] = name_slot
                    break

        response['belief_state'] = distribution_dict
        response['last_sys_act'] = str(sys_act)
        response['generated'] = str(nl)

        return response
