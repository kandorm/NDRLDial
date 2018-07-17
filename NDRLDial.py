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

        print '\n===============  Test Alter  =======================\n'

        # statistical data for all dial
        stats = {
            'informable': {
                'food': [10e-9, 10e-4, 10e-4, 10e-4],
                'pricerange': [10e-9, 10e-4, 10e-4, 10e-4],
                'area': [10e-9, 10e-4, 10e-4, 10e-4],
            },
            'requestable': {
                'food': [10e-9, 10e-4, 10e-4, 10e-4],
                'pricerange': [10e-9, 10e-4, 10e-4, 10e-4],
                'area': [10e-9, 10e-4, 10e-4, 10e-4],
                'phone': [10e-9, 10e-4, 10e-4, 10e-4],
                'address': [10e-9, 10e-4, 10e-4, 10e-4],
                'postcode': [10e-9, 10e-4, 10e-4, 10e-4],
                'name': [10e-9, 10e-4, 10e-4, 10e-4]
            },
            'vmc': 0.0, 'success': 0.0, 'turn_size': 0.0, 'alter': 0.0}

        log = {'belief_state': {}, 'stats': {}, 'result': []}

        for idx, dial in enumerate(self.corpus):
            print idx, '/', len(self.corpus)

            transcripts, state, goal, finished = dial
            constraint = copy.deepcopy(goal)
            del constraint['request']

            # data for one dial
            venue_name = ''
            venue_has_find = False
            turn = 0
            req = []
            recommended_list = []

            # for log
            c_log = {'dial': [], 'rec_list': [], 'goal': goal, 'finished': finished, 'venue': '',
                     'vmc': 0.0, 'success': 0.0, 'turn_size': 0.0, 'alter': 0.0}

            # param for dialogue continue
            belief_state = {}
            last_sys_act = ''
            for t_id, user_utt in enumerate(transcripts):
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

                c_dial = {'user_transcript': user_utt,
                          'true_state': str(state[t_id]),
                          'belief_state': str(SummaryUtils.getTopBeliefs(belief_state)),
                          'request': str(SummaryUtils.getRequestedSlots(belief_state)),
                          'sys_act': str(sys_act)}
                c_log['dial'].append(c_dial)

                t_state = state[t_id]
                for inf_slot in stats['informable']:
                    top_value, top_prob = SummaryUtils.getTopBelief(belief_state[inf_slot])
                    if inf_slot in t_state:
                        if t_state[inf_slot] == top_value:      # true positive
                            stats['informable'][inf_slot][0] += 1.0
                        else:                                   # false negative
                            stats['informable'][inf_slot][1] += 1.0
                    else:
                        if top_value == 'none':                 # true negative
                            stats['informable'][inf_slot][2] += 1.0
                        else:                                   # false positive
                            stats['informable'][inf_slot][3] += 1.0

                for req_slot in stats['requestable']:
                    t_req = t_state['request']
                    b_req = SummaryUtils.getRequestedSlots(belief_state)
                    if req_slot in t_req:
                        if req_slot in b_req:                   # true positive
                            stats['requestable'][req_slot][0] += 1.0
                        else:                                   # false negative
                            stats['requestable'][req_slot][1] += 1.0
                    else:
                        if req_slot not in b_req:               # true negative
                            stats['requestable'][req_slot][2] += 1.0
                        else:                                   # false positive
                            stats['requestable'][req_slot][3] += 1.0

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

        #################################################################
        # Compute req/inf pre recall ac
        quota = ['precision', 'recall', 'F-1', 'accuracy']
        inf_slots = ['food', 'pricerange', 'area']
        req_slots = ['food', 'pricerange', 'area', 'phone', 'address', 'postcode', 'name']
        stat_result = {'informable': {}, 'requestable': {}}
        for inf_slot in inf_slots:
            stat_result['informable'][inf_slot] = {}
            for c_q in quota:
                stat_result['informable'][inf_slot][c_q] = None
        stat_result['informable']['joint'] = {}

        for req_slot in req_slots:
            stat_result['requestable'][req_slot] = {}
            for c_q in quota:
                stat_result['requestable'][req_slot][c_q] = None
        stat_result['requestable']['joint'] = {}

        joint = [0.0 for x in range(4)]
        for s in inf_slots:
            joint = [joint[i] + stats['informable'][s][i] for i in range(len(joint))]
            tp, fn, tn, fp = stats['informable'][s]
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            ac = (tp + tn) / (tp + tn + fp + fn)
            stat_result['informable'][s]['precision'] = round(p, 4)
            stat_result['informable'][s]['recall'] = round(r, 4)
            stat_result['informable'][s]['F-1'] = round(2 * p * r / (p + r), 4)
            stat_result['informable'][s]['accuracy'] = round(ac, 4)
        tp, fn, tn, fp = joint
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        ac = (tp + tn) / (tp + tn + fp + fn)
        stat_result['informable']['joint']['precision'] = round(p, 4)
        stat_result['informable']['joint']['recall'] = round(r, 4)
        stat_result['informable']['joint']['F-1'] = round(2 * p * r / (p + r), 4)
        stat_result['informable']['joint']['accuracy'] = round(ac, 4)

        joint = [0.0 for x in range(4)]
        for s in req_slots:
            joint = [joint[i] + stats['requestable'][s][i] for i in range(len(joint))]
            tp, fn, tn, fp = stats['requestable'][s]
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            ac = (tp + tn) / (tp + tn + fp + fn)
            stat_result['requestable'][s]['precision'] = round(p, 4)
            stat_result['requestable'][s]['recall'] = round(r, 4)
            stat_result['requestable'][s]['F-1'] = round(2 * p * r / (p + r), 4)
            stat_result['requestable'][s]['accuracy'] = round(ac, 4)
        tp, fn, tn, fp = joint
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        ac = (tp + tn) / (tp + tn + fp + fn)
        stat_result['requestable']['joint']['precision'] = round(p, 4)
        stat_result['requestable']['joint']['recall'] = round(r, 4)
        stat_result['requestable']['joint']['F-1'] = round(2 * p * r / (p + r), 4)
        stat_result['requestable']['joint']['accuracy'] = round(ac, 4)

        log['belief_state'] = stat_result
        #################################################################

        print 'Average Turn Size : %.1f' % (stats['turn_size'] / float(len(self.corpus)))
        print 'Venue Match Rate  : %.1f%%' % (100 * stats['vmc'] / float(len(self.corpus)))
        print 'Task Success Rate : %.1f%%' % (100 * stats['success'] / float(len(self.corpus)))
        print 'Recommend one more: %.1f%%' % (100 * stats['alter'] / float(len(self.corpus)))

        log['stats']['vmc'] = str(round(100 * stats['vmc'] / float(len(self.corpus)), 2))
        log['stats']['success'] = str(round(100 * stats['success'] / float(len(self.corpus)), 2))
        log['stats']['turn_size'] = str(round(stats['turn_size'] / float(len(self.corpus)), 2))
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

        distribution_dict['entity'] = len(entities)

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
