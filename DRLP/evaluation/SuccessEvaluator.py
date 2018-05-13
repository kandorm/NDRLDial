import copy
from DRLP.utils import Settings
from DRLP.utils.dact import DiaAct, DiaActItem
from DRLP.ontology import Ontology
from KB import KnowledgeBase
import numpy as np


class Evaluator(object):

    def __init__(self):
        self.outcome = False
        self.num_turns = 0
        self.total_reward = 0

        self.rewards = []
        self.outcomes = []
        self.turns = []

    def get_turn_reward(self, turn_info):
        reward = self._get_turn_reward(turn_info)
        self.total_reward += reward
        self.num_turns += 1
        return reward

    def get_final_reward(self, final_info):
        if self.num_turns > 0:
            final_reward = self._get_final_reward(final_info)
            self.total_reward += final_reward

            self.rewards.append(self.total_reward)
            self.outcomes.append(self.outcome)
            self.turns.append(self.num_turns)
        else:
            final_reward = 0
        return final_reward

    def restart(self):
        self.num_turns = 0
        self.total_reward = 0

    ######################################################################
    # interface methods
    ######################################################################
    def _get_turn_reward(self, turn_info):
        """
        Computes the turn reward using turnInfo.

        Should be overridden by sub-class if values others than 0 should be returned.

        :param turn_info: parameters necessary for computing the turn reward, eg., system act, prev system act and state
        :type turn_info: dict
        :return: int -- the turn reward, default 0.
        """
        return 0

    def _get_final_reward(self, final_info):
        """
        Computes the final reward using final_info and sets the dialogue outcome.

        Should be overridden by sub-class if values others than 0 should be returned.

        :param final_info: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type final_info: dict
        :return: int -- the final reward, default 0.
        """
        return 0

    def do_training(self):
        """
        Defines whether the currently evaluated dialogue should be used for training.

        Should be overridden by sub-class if values others than True should be returned.

        :return: bool -- whether the dialogue should be used for training
        """
        return True

    ######################################################################
    # print methods
    ######################################################################
    def print_summary(self):
        """
        Prints the summary of a run - ie multiple dialogs. Assumes dialogue outcome represents success. For other types, override methods in sub-class.
        """
        num_dialogs = len(self.rewards)
        assert (len(self.outcomes) == num_dialogs)
        assert (len(self.turns) == num_dialogs)

        # computing t-value for confidence interval of 95%
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)

        print 'evaluated by::', type(self).__name__
        print '# of dialogues  = %d' % num_dialogs
        if num_dialogs:
            print 'Average reward  = %.2f +- %.2f' % (np.mean(self.rewards),
                                                      tinv * np.std(self.rewards) / np.sqrt(num_dialogs))
            print self._getResultString(self.outcomes)
            print 'Average turns   = %.2f +- %.2f' % (np.mean(self.turns),
                                                      tinv * np.std(self.turns) / np.sqrt(num_dialogs))

    def _getResultString(self, outcomes):
        num_dialogs = len(outcomes)
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)
        return 'Average success = {0:0.2f} +- {1:0.2f}'.format(100 * np.mean(outcomes),
                                                               100 * tinv * np.std(outcomes) / np.sqrt(num_dialogs))


class ObjectiveSuccessEvaluator(Evaluator):

    def __init__(self):
        super(ObjectiveSuccessEvaluator, self).__init__()

        self.penalise_all_turns = True  # We give -1 each turn. Note that this is done thru this boolean
        self.reward_venue_recommended = 0
        self.reward_venue_alternatives = 10
        self.penalty_venue_alternatives = 5
        self.wrong_venue_penalty = 0
        self.not_mentioned_value_penalty = 0
        self.successReward = 20
        self.failPenalty = 0

        if Settings.config.has_option('eval', 'penaliseallturns'):
            self.penalise_all_turns = Settings.config.getboolean('eval', 'penaliseallturns')
        if Settings.config.has_option('eval', 'rewardvenuerecommended'):
            self.reward_venue_recommended = Settings.config.getint('eval', 'rewardvenuerecommended')
        if Settings.config.has_option('eval', 'rewardvenuealternatives'):
            self.reward_venue_alternatives = Settings.config.getint('eval', 'rewardvenuealternatives')
        if Settings.config.has_option('eval', 'penaltyvenuealternatives'):
            self.penalty_venue_alternatives = Settings.config.getint('eval', 'penaltyvenuealternatives')
        if Settings.config.has_option('eval', 'wrongvenuepenalty'):
            self.wrong_venue_penalty = Settings.config.getint('eval', 'wrongvenuepenalty')
        if Settings.config.has_option('eval', 'notmentionedvaluepenalty'):
            self.not_mentioned_value_penalty = Settings.config.getint('eval', 'notmentionedvaluepenalty')
        if Settings.config.has_option("eval", "successreward"):
            self.successReward = Settings.config.getint("eval", "successreward")
        if Settings.config.has_option("eval", "failpenalty"):
            self.failPenalty = Settings.config.getint("eval", "failpenalty")

        self.user_goal = None
        self.venue_recommended = False
        self.last_venue_recommend = None
        self.DM_history = None
        self.mentioned_values = {}  # {slot: set(values), ...}
        sys_req_slots = Ontology.global_ontology.get_system_requestable_slots()
        for slot in sys_req_slots:
            self.mentioned_values[slot] = set(['dontcare'])

    def restart(self):
        super(ObjectiveSuccessEvaluator, self).restart()
        self.venue_recommended = False
        self.last_venue_recommend = None

    def _get_turn_reward(self, turn_info):
        """
        Computes the turn reward regarding turn_info. The default turn reward is -1 unless otherwise computed.

        :param turn_info: parameters necessary for computing the turn reward, eg., system act, prev system act and state
        :type turn_info: dict
        :return: int -- the turn reward.
        """
        # Immediate reward for each turn.
        reward = -self.penalise_all_turns

        if turn_info is not None and isinstance(turn_info, dict):
            if 'usermodel' in turn_info and 'sys_act' in turn_info:
                um = turn_info['usermodel']
                self.user_goal = um.goal.constraints

                # unpack input user model um.
                prev_consts = copy.deepcopy(um.goal.constraints)
                for item in prev_consts:
                    if item.slot == 'name' and item.op == '=':
                        item.val = 'dontcare'
                requests = um.goal.requests
                sys_act = DiaAct(turn_info['sys_act'])
                user_act = um.lastUserAct

                # Check if the most recent venue satisfies constraints.
                name = sys_act.get_value('name', negate=False)
                if name not in ['none', None]:
                    # Venue is recommended.
                    is_valid_venue = self._is_valid_venue(name, prev_consts)
                    if is_valid_venue:
                        # Success except if the next user action is reqalts.
                        if user_act.act != 'reqalts':
                            self.venue_recommended = True  # Correct venue is recommended.
                        else:
                            if self.last_venue_recommend != name:
                                reward += self.reward_venue_alternatives
                            else:
                                reward -= self.penalty_venue_alternatives
                    else:
                        # Previous venue did not match.
                        self.venue_recommended = False
                        reward -= self.wrong_venue_penalty

                    self.last_venue_recommend = name

                # If system inform(name=none) but it was not right decision based on wrong values.
                if name == 'none' and sys_act.has_conflicting_value(prev_consts):
                    reward -= self.wrong_venue_penalty

                # Check if the system used slot values previously not mentioned for 'select' and 'confirm'.
                not_mentioned = False
                if sys_act.act in ['select', 'confirm']:
                    for slot in Ontology.global_ontology.get_system_requestable_slots():
                        values = set(sys_act.get_values(slot))
                        if len(values - self.mentioned_values[slot]) > 0:
                            # System used values which are not previously mentioned.
                            not_mentioned = True
                            break

                if not_mentioned:
                    reward -= self.not_mentioned_value_penalty

                # If the correct venue has been recommended and all requested slots are filled,
                # check if this dialogue is successful.
                if self.venue_recommended and None not in requests.values():
                    reward += self.reward_venue_recommended

                # Update mentioned values.
                self._update_mentioned_value(sys_act)
                self._update_mentioned_value(user_act)

        return reward

    def _get_final_reward(self, final_info):
        if final_info is not None and isinstance(final_info, dict):
            if 'usermodel' in final_info:    # from user simulator
                um = final_info['usermodel']
                if um is None:
                    self.outcome = False
                else:
                    requests = um.goal.requests
                    if None not in requests.values():
                        valid_venue = self._is_valid_venue(requests['name'], self.user_goal)
                        if valid_venue:
                            self.outcome = True

        return self.outcome * self.successReward - (not self.outcome) * self.failPenalty

    def _update_mentioned_value(self, act):

        sys_req_slots = Ontology.global_ontology.get_system_requestable_slots()
        for item in act.items:
            if item.slot in sys_req_slots and item.val not in [None, 'none']:
                self.mentioned_values[item.slot].add(item.val)

    def _is_valid_venue(self, name, constraints):
        constraints2 = None
        if isinstance(constraints, list):
            constraints2 = copy.deepcopy(constraints)
            for const in constraints2:
                if const.slot == 'name':
                    if const.op == '!=':
                        if name == const.val and const.val != 'dontcare':
                            return False
                        else:
                            constraints2.remove(const)
                    elif const.op == '=':
                        if name != const.val and const.val != 'dontcare':
                            return False
            constraints2.append(DiaActItem('name', '=', name))
        elif isinstance(constraints, dict):  # should never be the case, um has DActItems as constraints
            constraints2 = copy.deepcopy(constraints)
            for slot in constraints:
                if slot == 'name' and name != constraints[slot]:
                    return False
            constraints2['name'] = name
        entities = KnowledgeBase.global_kb.entity_by_features(constraints2)

        return any(entities)
