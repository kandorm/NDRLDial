import copy
from DRLP.utils import Settings
from DRLP.utils.dact import DiaActItem, DiaAct
from DRLP.ontology import Ontology
from DRLP.usersimulator.UMHdcSim import UMHdcSim
from KB import KnowledgeBase


class UMGoal(object):

    def __init__(self, patience):
        self.constraints = []
        self.requests = {}
        self.prev_slot_values = {}
        self.patience = patience
        self.request_type = Ontology.global_ontology.get_type()

        self.system_has_informed_name_none = False
        self.no_relaxed_constraints_after_name_none = False

    def clear(self, patience):
        self.constraints = []
        self.requests = {}
        self.prev_slot_values = {}
        self.patience = patience
        self.request_type = Ontology.global_ontology.get_type()

        self.system_has_informed_name_none = False
        self.no_relaxed_constraints_after_name_none = False

    def add_request(self, slot):
        self.requests[slot] = None

    '''
    Methods for constraints.
    '''
    def add_const(self, slot, value, negate=False):
        if not negate:
            op = '='
        else:
            op = '!='
        item = DiaActItem(slot, op, value)
        self.constraints.append(item)

    def replace_const(self, slot, value, negate=False):
        self.remove_slot_const(slot, negate)
        self.add_const(slot, value, negate)

    def contains_slot_const(self, slot):
        for item in self.constraints:
            if str(item.slot) == slot:
                return True
        return False

    def remove_slot_const(self, slot, negate=None):
        copy_consts = copy.deepcopy(self.constraints)

        if negate is not None:
            if not negate:
                op = '='
            else:
                op = '!='

            for item in copy_consts:
                if item.slot == slot:
                    if item.op == op:
                        self.constraints.remove(item)
        else:
            for item in copy_consts:
                if item.slot == slot:
                    self.constraints.remove(item)

    def get_correct_const_value(self, slot, negate=False):
        """
        :return: (list of) value of the given slot in user goal constraint.
        """
        values = []
        for item in self.constraints:
            if item.slot == slot:
                if item.op == '!=' and negate or item.op == '=' and not negate:
                    values.append(item.val)

        if len(values) == 1:
            return values[0]
        elif len(values) == 0:
            return None
        print 'Multiple values are found for %s in constraint: %s' % (slot, str(values))
        return values

    def get_correct_const_value_list(self, slot, negate=False):
        """
        :return: (list of) value of the given slot in user goal constraint.
        """
        values = []
        for item in self.constraints:
            if item.slot == slot:
                if (item.op == '!=' and negate) or (item.op == '=' and not negate):
                    values.append(item.val)
        return values

    def add_prev_used(self, slot, value):
        """
        Adds the given slot-value pair to the record of previously used slot-value pairs.
        """
        if slot not in self.prev_slot_values:
            self.prev_slot_values[slot] = set()
        self.prev_slot_values[slot].add(value)

    def add_name_constraint(self, value, negate=False):
        if value in [None, 'none']:
            return

        wrong_venues = self.get_correct_const_value_list('name', negate=True)
        correct_venue = self.get_correct_const_value('name', negate=False)

        if not negate:
            # Adding name=value but there is name!=value.
            if value in wrong_venues:
                print 'Failed to add name=%s: already got constraint name!=%s.' % (value, value)
                return
            # Can have only one name= constraint.
            if correct_venue is not None:
                self.replace_const('name', value)
                return

            # Adding name=value, then remove all name!=other.
            self.replace_const('name', value)
            return

        if negate:
            # Adding name!=value but there is name=value.
            if correct_venue == value:
                print 'Failed to add name!=%s: already got constraint name=%s.' % (value, value)
                return
            # Adding name!=value, but there is name=other. No need to add.
            if correct_venue is not None:
                return

            self.add_const('name', value, negate=True)
            return

    def is_satisfy_all_consts(self, item):
        """
        Check if all the given items set[(slot, op, value),..]
        satisfies all goal constraints (conjunction of constraints).
        """
        if type(item) is not set:
            item = set([item])
        for it in item:
            for const in self.constraints:
                if not const.match(it):
                    return False
        return True

    def is_completed(self):
        # If the user has not specified any constraints, return True
        if not self.constraints:
            return True
        if (self.system_has_informed_name_none and not self.no_relaxed_constraints_after_name_none) or\
                (self.is_venue_recommended() and self.are_all_requests_filled()):
            return True
        return False

    '''
    Methods for requests.
    '''
    def reset_requests(self):
        for info in self.requests:
            self.requests[info] = None

    def fill_requests(self, dact_items):
        for item in dact_items:
            if item.op != '!=':
                self.requests[item.slot] = item.val

    def are_all_requests_filled(self):
        """
        Returns True if all request slots have a non-empty value.
        """
        return None not in self.requests.values()

    def is_venue_recommended(self):
        """
        Returns True if the request slot 'name' is not empty.
        :return:
        """
        if 'name' in self.requests and self.requests['name'] is not None:
            return True
        return False

    def get_unsatisfied_requests(self):
        results = []
        for info in self.requests:
            if self.requests[info] is None:
                results.append(info)
        return results

    def __str__(self):
        result = 'constraints: ' + str(self.constraints) + '\n'
        result += 'requests:    ' + str(self.requests) + '\n'
        if self.patience is not None:
            result += 'patience:    ' + str(self.patience) + '\n'
        return result


class GoalGenerator(object):

    def __init__(self):
        self.MIN_VENUES_PER_GOAL = 1
        self.MAX_VENUES_PER_GOAL = 3
        self.MIN_REQUESTS = 1
        self.MAX_REQUESTS = 3

    def init_goal(self, um_patience):
        goal = UMGoal(um_patience)
        num_attempts_to_resample = 2000
        while True:
            num_attempts_to_resample -= 1
            # Randomly sample a goal (ie constraints):
            self._init_consts_requests(goal, um_patience)
            # Check that there are venues that satisfy the constraints:
            venues = KnowledgeBase.global_kb.entity_by_features(goal.constraints)
            if self.MIN_VENUES_PER_GOAL < len(venues) < self.MAX_VENUES_PER_GOAL:
                break
        return goal

    def _init_consts_requests(self, goal, um_patience):
        """
        Randomly initialises constraints and requests of the given goal.
        """
        goal.clear(um_patience)
        # ========================  Setting informable slots and values  =========================================
        # Get a list of informable slots.
        valid_constraint_slots = Ontology.global_ontology.get_system_requestable_slots()    # system_requestable
        # Randomly sample some slots from those that are valid:
        random_slots = list(Settings.random.choice(valid_constraint_slots,
                                                   size=min(self.MAX_VENUES_PER_GOAL, len(valid_constraint_slots)),
                                                   replace=False))
        # Randomly fill in some constraints for the sampled slots:
        for slot in random_slots:
            goal.add_const(slot, Ontology.global_ontology.get_random_value_for_slot(slot, no_dontcare=False))

        # ========================  Setting requestable slots  =========================================
        # Add requests. Assume that the user always wants to know the name of the place
        goal.add_request('name')

        if self.MIN_REQUESTS == self.MAX_REQUESTS:
            n = self.MIN_REQUESTS - 1   # since 'name' is already included
        else:
            n = Settings.random.randint(low=self.MIN_REQUESTS - 1, high=self.MAX_REQUESTS)
        valid_req_slots = Ontology.global_ontology.get_user_requestable_slots()
        if len(valid_req_slots) >= n > 0:
            choices = Settings.random.choice(valid_req_slots, n, replace=False)
            for req_slot in choices:
                goal.add_request(req_slot)


class UserModel(object):

    def __init__(self):
        self.max_patience = 5
        self.patience_old_style = False

        self.goal = None
        self.prev_goal = None
        self.lastUserAct = None
        self.lastSysAct = None

        self.generator = GoalGenerator()
        self.hdcSim = UMHdcSim()

    def init(self):
        self.lastUserAct = None
        self.lastSysAct = None
        self.goal = self.generator.init_goal(self.max_patience)
        self.hdcSim.init(self.goal, self.max_patience)

    def receive(self, sys_act):
        """
        This method is called to transmit the machine dialogue act to the user.
        It updates the goal and the agenda.
        :param sys_act: System action.
        :return:
        """
        # Update previous goal.
        self.prev_goal = copy.deepcopy(self.goal)
        # Update the user patience level.
        if self.lastUserAct is not None and self.lastUserAct.act == 'repeat' and \
                self.lastSysAct is not None and self.lastSysAct.act == 'repeat' and \
                sys_act.act == 'repeat':
            # Endless cycle of repeating repeats: reduce patience to zero.
            print "Endless cycle of repeating repeats. Setting patience to zero."
            self.goal.patience = 0
        elif sys_act.act == 'badact' or sys_act.act == 'null' or \
                (self.lastSysAct is not None and self.lastUserAct.act != 'repeat' and self.lastSysAct == sys_act):
            # Same action as last turn. Patience decreased.
            self.goal.patience -= 1
        elif self.patience_old_style:
            # not same action as last time so patience is restored
            self.goal.patience = self.max_patience

        if self.goal.patience < 1:
            self.hdcSim.agenda.clear()
            # Pushing bye act onto agenda.
            self.hdcSim.agenda.push(DiaAct('bye()'))
            return

        # Update last system action.
        self.lastSysAct = sys_act

        # Process the sys_act
        self.hdcSim.receive(sys_act, self.goal)

    def respond(self):
        """
        This method is called after receive() to get the user dialogue act response.
        The method first increments the turn counter, then pops n items off the agenda to form
        the response dialogue act. The agenda and goal are updated accordingly.
        :returns: (instance) of :class:`DiaAct`
        """
        user_output = self.hdcSim.respond(self.goal)

        if user_output.act == 'request' and len(user_output.items) > 0:
            if user_output.contains_slot('near') and self.goal.contains_slot_const('near'):
                for const in self.goal.constraints:
                    if const.slot == 'near':
                        near_const = const.val
                        near_op = const.op
                        break
                # original: bug. constraints is a list --- near_const = self.goal.constraints['near']
                if near_const != 'dontcare':
                    if near_op == "=":  # should be true for 'dontcare' value
                        user_output.act = 'confirm'
                        user_output.items[0].val = near_const

        self.lastUserAct = user_output
        # self.goal.update_au(user_output)
        return user_output
