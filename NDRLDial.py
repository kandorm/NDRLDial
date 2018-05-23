import copy
import json
from loader.DataReader import DataReader
from DRLP.policy import SummaryUtils
from utils.dact import DiaAct
from NBT.tracker.net import Tracker
from DRLP.Model import DRLP
from KB.KBManager import KBManager
from SemI.RegexSemI import RegexSemI
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

        self.semi = RegexSemI()
        self.policy_manager = DRLP(self.policy_config)
        self.tracker = Tracker(self.tracker_config)
        # Connect to the knowledge base
        self.kb_manager = KBManager()
        self.generator = BasicSemO()

        self.ontology_file = parser.get('file', 'ontology')
        self.corpus_file = parser.get('file', 'corpus')
        self.result_file = parser.get('file', 'result')
        self.reader = DataReader(ontology_file=self.ontology_file, corpus_file=self.corpus_file)

        self.ontology = self.reader.ontology
        self.corpus = self.reader.corpus

        self.requestable_slots = self.ontology['requestable']
        self.user_intent = self.ontology['user_intent']

    def test(self):

        print '\n===============  Test Model  =======================\n'

        stats = {'vmc': 0.0, 'success': 0.0, 'turn_size': 0.0, 'belief_match': 0.0}
        log = {'stats': {}, 'result': []}

        for idx, dial in enumerate(self.corpus):

            print idx, '/', len(self.corpus)

            transcripts, goal, finished = dial

            constraint = copy.deepcopy(goal)
            del constraint['request']

            venue_name = ''
            venue_has_find = False
            turn = 0
            req = []
            belief_request = []

            # for log
            c_log = {'dial': [], 'goal': goal, 'finished': finished, 'venue': '',
                     'vmc': 0.0, 'success': 0.0, 'turn_size': 0.0, 'belief_match': 0.0}

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
                        if item.slot == 'name' and item.op == '=' and item.val not in ['none', 'dontcare', None]:
                            venue_name = item.val
                            if not venue_has_find:
                                constraint['name'] = venue_name
                                entities = self.kb_manager.entity_by_features(constraint)
                                if len(entities) > 0:
                                    stats['turn_size'] += turn
                                    c_log['turn_size'] += turn

                        if item.slot in self.requestable_slots:
                            req.append(item.slot)

                belief_request.extend(belief_state['request'])

                c_log['dial'].append({'user_transcript': user_utt,
                                      'belief_state': str(SummaryUtils.getTopBeliefs(belief_state)),
                                      'request': str(SummaryUtils.getRequestedSlots(belief_state)),
                                      'sys_act': str(sys_act)})

            #################################################################
            # Test belief state match
            belief_state = SummaryUtils.getTopBeliefs(belief_state)
            b_match = True
            for slot in goal.keys():
                if slot == 'request':
                    if set(goal[slot]) & set(belief_request) != set(goal[slot]):
                        b_match = False
                else:
                    try:
                        if goal[slot] != belief_state[slot][0]:
                            b_match = False
                    except KeyError:
                        b_match = False
            if b_match:
                c_log['belief_match'] += 1.0
                stats['belief_match'] += 1.0
            #################################################################

            c_log['venue'] = venue_name
            if not venue_has_find:
                stats['turn_size'] += turn
                c_log['turn_size'] += turn

            if venue_name and finished:
                constraint['name'] = venue_name
                entities = self.kb_manager.entity_by_features(constraint)
                if len(entities) > 0:
                    stats['vmc'] += 1
                    c_log['vmc'] += 1
                    if set(req) & set(goal['request']) == set(goal['request']):
                        stats['success'] += 1
                        c_log['success'] += 1

            log['result'].append(c_log)

        print 'Venue Match Rate  : %.1f%%' % (100 * stats['vmc'] / float(len(self.corpus)))
        print 'Task Success Rate : %.1f%%' % (100 * stats['success'] / float(len(self.corpus)))
        print 'Average Turn Size : %.1f' % (stats['turn_size'] / float(len(self.corpus)))
        print 'Belief state match: %.1f%%' % (100 * stats['belief_match'] / float(len(self.corpus)))

        log['stats']['vmc'] = str(round(100 * stats['vmc'] / float(len(self.corpus)), 2))
        log['stats']['success'] = str(round(100 * stats['success'] / float(len(self.corpus)), 2))
        log['stats']['turn_size'] = str(round(stats['turn_size'] / float(len(self.corpus)), 2))
        log['stats']['belief_match'] = str(round(100 * stats['belief_match'] / float(len(self.corpus)), 2))

        json.dump(log, open(self.result_file, "w"), indent=2)

    def test_alter(self):

        print '\n===============  Test Alter  =======================\n'

        # statistical data for all dial
        stats = {'vmc': 0.0, 'success': 0.0, 'turn_size': 0.0, 'belief_match': 0.0, 'alter': 0.0}
        log = {'stats': {}, 'result': []}

        for idx, dial in enumerate(self.corpus):
            print idx, '/', len(self.corpus)

            transcripts, goal, finished = dial
            constraint = copy.deepcopy(goal)
            del constraint['request']

            # data for one dial
            venue_name = ''
            venue_has_find = False
            turn = 0
            req = []
            belief_request = []
            recommended_list = []

            # for log
            c_log = {'dial': [], 'rec_list': [], 'goal': goal, 'finished': finished, 'venue': '',
                     'vmc': 0.0, 'success': 0.0, 'turn_size': 0.0, 'belief_match': 0.0, 'alter': 0.0}

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
                        if item.slot == 'name' and item.op == '=' and item.val not in ['none', 'dontcare', None]:
                            venue_name = item.val
                            recommended_list.append(venue_name)
                            if not venue_has_find:
                                constraint['name'] = venue_name
                                entities = self.kb_manager.entity_by_features(constraint)
                                if len(entities) > 0:
                                    venue_has_find = True
                                    stats['turn_size'] += turn
                                    c_log['turn_size'] += turn

                        if item.slot in self.requestable_slots:
                            req.append(item.slot)

                belief_request.extend(SummaryUtils.getRequestedSlots(belief_state))

                c_log['dial'].append({'user_transcript': user_utt,
                                      'belief_state': str(SummaryUtils.getTopBeliefs(belief_state)),
                                      'request': str(SummaryUtils.getRequestedSlots(belief_state)),
                                      'sys_act': str(sys_act)})

            #################################################################
            # Test belief state match
            belief_state = SummaryUtils.getTopBeliefs(belief_state)
            b_match = True
            for slot in goal.keys():
                if slot == 'request':
                    if set(goal[slot]) & set(belief_request) != set(goal[slot]):
                        b_match = False
                else:
                    try:
                        if goal[slot] != belief_state[slot][0]:
                            b_match = False
                    except KeyError:
                        b_match = False
            if b_match:
                c_log['belief_match'] += 1.0
                stats['belief_match'] += 1.0
            #################################################################

            #################################################################
            # Test recommend more than one restaurant
            recommended_list = list(set(recommended_list))
            c_log['rec_list'] = recommended_list
            if len(recommended_list) > 1:
                stats['alter'] += 1
                c_log['alter'] += 1
            #################################################################

            c_log['venue'] = venue_name
            if not venue_has_find:
                stats['turn_size'] += turn
                c_log['turn_size'] += turn

            if venue_name and finished:
                constraint['name'] = venue_name
                entities = self.kb_manager.entity_by_features(constraint)
                if len(entities) > 0:
                    stats['vmc'] += 1
                    c_log['vmc'] += 1
                    if set(req) & set(goal['request']) == set(goal['request']):
                        stats['success'] += 1
                        c_log['success'] += 1

            log['result'].append(c_log)

        print 'Average Turn Size : %.1f' % (stats['turn_size'] / float(len(self.corpus)))
        print 'Belief state match: %.1f%%' % (100 * stats['belief_match'] / float(len(self.corpus)))
        print 'Venue Match Rate  : %.1f%%' % (100 * stats['vmc'] / float(len(self.corpus)))
        print 'Task Success Rate : %.1f%%' % (100 * stats['success'] / float(len(self.corpus)))
        print 'Recommend one more: %.1f%%' % (100 * stats['alter'] / float(len(self.corpus)))

        log['stats']['vmc'] = str(round(100 * stats['vmc'] / float(len(self.corpus)), 2))
        log['stats']['success'] = str(round(100 * stats['success'] / float(len(self.corpus)), 2))
        log['stats']['turn_size'] = str(round(stats['turn_size'] / float(len(self.corpus)), 2))
        log['stats']['belief_match'] = str(round(100 * stats['belief_match'] / float(len(self.corpus)), 2))
        log['stats']['alter'] = str(round(100 * stats['alter'] / float(len(self.corpus)), 2))

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
            distribution_dict['name'] = []

        # 2.Query Knowledge base
        constraints = {}
        slots = self.ontology['system_requestable']
        for slot in slots:
            constraints[slot] = prediction_dict[slot]
        entities = self.kb_manager.entity_by_features(constraints)

        # 3. Add user intent into belief state
        intent_predicted = self.semi.decode(str(user_utt).lower())
        distribution_dict['user_intent'] = dict.fromkeys(self.user_intent, 0.0)
        for slot in intent_predicted:
            if slot in distribution_dict['user_intent']:
                distribution_dict['user_intent'][slot] = intent_predicted[slot]

        # 4. Policy -- Determine system act/response type: DiaAct
        sys_act = self.policy_manager.act_on(distribution_dict, entities)

        # 5. Generate natural language
        nl = self.generator.generate(str(sys_act))

        if not isinstance(sys_act, DiaAct):
            sys_act = DiaAct(str(sys_act))

        if sys_act.act == 'inform':
            name = sys_act.get_value('name', negate=False)
            if name not in ['none', None]:
                try:
                    distribution_dict['name'].remove(name)
                except:
                    pass
                distribution_dict['name'].append(name)

        response['belief_state'] = distribution_dict
        response['last_sys_act'] = str(sys_act)
        response['generated'] = str(nl)

        return response
