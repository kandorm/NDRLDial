import copy
import json
from loader.DataReader import DataReader
from DRLPolicy.policy import SummaryUtils
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
        self.result_file = parser.get('file', 'result')
        self.reader = DataReader(ontology_file=self.ontology_file, corpus_file=self.corpus_file)

        self.ontology = self.reader.ontology
        self.corpus = self.reader.corpus

        self.name_values = self.ontology['informable']['name'] + ['none']

    def test(self):

        print '\n===============  Test Model  =======================\n'

        stats = {'success': 0.0, 'turn_size': 0.0}

        log = {'stats': {}, 'result': []}

        for idx, dial in enumerate(self.corpus):

            print idx, '/', len(self.corpus)

            transcripts, goal, finished = dial

            constraint = copy.deepcopy(goal)
            del constraint['request']

            venue_name = ''
            venue_has_find = False
            turn = 0

            # for log
            c_log = {'dial': [], 'goal': goal, 'finished': finished, 'venue': '',
                     'success': 0.0, 'turn_size': 0.0}

            # param for dialogue continue
            belief_state = {}
            last_sys_act = ''
            for user_utt in transcripts:
                turn += 1
                response = self.reply(user_utt, last_sys_act, belief_state)
                belief_state = response['belief_state']
                last_sys_act = response['last_sys_act']

                sys_act = DiaAct(last_sys_act)
                if sys_act.act == 'inform':
                    for item in sys_act.items:
                        name = sys_act.get_value('name', negate=False)
                        if name not in ['none', None]:
                            venue_name = name
                            if not venue_has_find:
                                constraint['name'] = venue_name
                                entities = self.kb_manager.entity_by_features(constraint)
                                if len(entities) > 0:
                                    venue_has_find = True
                                    stats['turn_size'] += turn
                                    c_log['turn_size'] += turn

                c_log['dial'].append({'user_transcript': user_utt,
                                      'belief_state': str(SummaryUtils.getTopBeliefs(belief_state)),
                                      'sys_act': str(sys_act)})

            c_log['venue'] = venue_name

            if not venue_has_find:
                stats['turn_size'] += turn
                c_log['turn_size'] += turn

            if venue_name:
                constraint['name'] = venue_name
                entities = self.kb_manager.entity_by_features(constraint)
                if len(entities) > 0:
                    stats['success'] += 1
                    c_log['success'] += 1

            log['result'].append(c_log)

        print 'Task Success Rate : %.1f%%' % (100 * stats['success'] / float(len(self.corpus)))
        print 'Average Turn Size : %.1f' % (stats['turn_size'] / float(len(self.corpus)))

        log['stats']['success'] = str(round(100 * stats['success'] / float(len(self.corpus)), 2))
        log['stats']['turn_size'] = str(round(stats['turn_size'] / float(len(self.corpus)), 2))

        json.dump(log, open(self.result_file, "w"), indent=2)

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
