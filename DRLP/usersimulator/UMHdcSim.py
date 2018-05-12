import copy
from DRLP.utils.dact import DiaAct, DiaActItem
from DRLP.utils import Settings
from DRLP.ontology import Ontology

'''
User Acts
 -- request(slot)
 -- inform(slot=value)
 -- confirm(slot=value)
 -- hello()
 -- negate(slot=value)
 -- deny(slot1=value1, slot1=value2)
 -- bye()
'''

'''
System Acts
 -- request(slot)
 -- inform(slot=value, ...)
 -- confirm(slot=value)
 -- hello()
 -- bye()
 -- confreq(slot=value,slot)
'''


class UMAgenda(object):

    def __init__(self):
        self.agenda_items = []  # Stack of DiaAct

    def init(self, goal):
        self.agenda_items = []
        self.push_front(DiaAct('inform(type="%s")' % goal.request_type))

        for const in goal.constraints:
            slot = const.slot
            if slot == 'method':
                continue

            dia_act = DiaAct('inform()')
            val = goal.get_correct_const_value(slot)
            if val is not None:
                dia_act.append(slot, val)
            if not self.contains(slot, val):
                self.push_front(dia_act)

        self.push_front(DiaAct('bye()'))

    def contains(self, slot, value, negate=False):
        """
        Check if slot, value pair is in an agenda dialogue act
        :param slot:
        :param value:
        :param negate: None
        :type negate: bool
        :returns: (bool) slot, value is in an agenda dact?
        """
        for dact in self.agenda_items:
            if dact.contains(slot, value, negate):
                return True
        return False

    def get_agenda_with_act(self, act):
        """
        agenda items with this act
        :param act: dialogue act
        :type act: str
        :return: (list) agenda items
        """
        items = []
        for ait in self.agenda_items:
            if ait.act == act:
                items.append(ait)
        return items

    def get_agenda_with_act_slot(self, act, slot):
        """
        :param act: dialogue act
        :type act: str
        :param slot: slot name
        :type slot: str
        :return: (list) of agenda items
        """
        items = []
        for ait in self.agenda_items:
            if ait.act == act:
                for item in ait.items:
                    if item.slot == slot:
                        items.append(ait)
                        break
        return items

    def filter_acts_slot(self, slot):
        """
        Any acts related to the given slot are removed from the agenda.
        :param slot: slot name
        :type slot: str
        :return: None
        """
        deleted = []
        for ait in self.agenda_items:
            if ait.act in ['inform', 'confirm', 'affirm'] and len(ait.items) > 0:
                if len(ait.items) > 1:
                    pass
                for only_item in ait.items:
                    if only_item.slot == slot:
                        deleted.append(ait)

        for ait in deleted:
            self.agenda_items.remove(ait)

    def filter_constraints(self, dap):
        """
        Filters out acts on the agenda that convey the constraints mentioned in the given dialogue act.
        Calls :meth:`filter_acts_slot` to do so.
        :param dap:
        :returns: None
        """
        if dap.act in ['inform', 'confirm'] or \
                (dap.act in ['affirm', 'negate'] and len(dap.items) > 0):
            for item in dap.items:
                self.filter_acts_slot(item.slot)

    def size(self):
        """
        Utility func to get size of agenda_items list
        :returns: (int) length
        """
        return len(self.agenda_items)

    def clear(self):
        """
        Erases all acts on the agenda (empties list)
        :return: None
        """
        self.agenda_items = []

    def push_front(self, dia_act):
        """
        Adds the given dialogue act to the front of the agenda
        :param dia_act:
        :returns: None
        """
        self.agenda_items = [dia_act] + self.agenda_items

    def push(self, dia_act):
        self.agenda_items.append(dia_act)

    def pop(self):
        return self.agenda_items.pop()

    def remove(self, dia_act):
        self.agenda_items.remove(dia_act)


class UMHdcSim(object):
    """
    Handcrafted behaviour of simulated user
    """
    def __init__(self, max_patience=5):
        self.max_patience = max_patience

        # DEFAULTS:
        self.answer_req_always = True
        self.use_new_goal_scenarios = False
        self.sampleDecisiconProbs = False
        self.patience_old_style = False
        self.old_style_parameter_sampling = True
        self.rand_decision_probs = {'InformCombination': 0.6,
                                    'AddSlotToReq': 0.333,
                                    'NoAfterReqmore': 0.7,
                                    'YesAfterReqmore': 0.25,
                                    'Greeting': 0.5,
                                    'ConstraintRelax': 0.667,
                                    'TellAboutChange': 0.5,
                                    'ByeOrStartOver': 0.333,
                                    'DealWithPending': 0.5,
                                    'AddVenueNameToRequest': 0.05,
                                    'NoSlotWithDontcare': 0.8,
                                    'Repeat': 0.0,
                                    'InformToConfirm': 0.05,
                                    'CombAffirmWithAgdItem': 0.05,
                                    'NullResp': 0.0,
                                    'OverruleCorrection': 0.1,
                                    'ConfirmRandomConstr': 0.1,
                                    'ReqAltsAfterVenRec1': 0.143,
                                    'ReqAltsAfterVenRec2': 0.143,
                                    'NewRequestResp1': 0.2,
                                    'NewRequestResp2': 0.2,
                                    'CorrectingAct1': 0.45,
                                    'CorrectingAct2': 0.4,
                                    'ThankAck1': 0.1,
                                    'ThankAck2': 0.1,
                                    'AffirmCombination': 1.0}

        self.receive_options = {'badact': self._receive_badact,
                                'reqmore': self._receive_reqmore,
                                'null': self._receive_badact,
                                'hello': self._receive_hello,
                                'request': self._receive_request,
                                'confirm': self._receive_confirm,
                                'inform': self._receive_inform,
                                'repeat': self._receive_repeat,
                                'select': self._receive_select,
                                'confreq': self._receive_confreq,
                                'bye': self._receive_bye,
                                'affirm': self._receive_affirm,
                                'negate': self._receive_negate}

        self.agenda = UMAgenda()
        self.last_user_act = None
        self.last_sys_act = None

        self.relax_constraints = None
        self.first_venue_recommendation = None

    def init(self, goal, um_patience):
        # Create the agenda
        self.agenda.init(goal)

        self.last_user_act = DiaAct('null()')
        self.last_sys_act = DiaAct('null()')
        self.max_patience = um_patience

        self.relax_constraints = False
        self.first_venue_recommendation = True

    def receive(self, sys_act, goal):
        self.last_sys_act = sys_act
        # If agenda is empty, push ByeAct on top.
        if goal.is_completed() and self.agenda.size() == 0 and sys_act.dact['act'] != 'reqmore'\
                and Settings.random.rand() < 0.85:
            self.agenda.clear()
            self.agenda.push(DiaAct('bye()'))
            return

        # Generate repeat act with small probability:
        #   assume the user did not hear the system utterance,
        #   let alone make any updates to their (user goal) state,
        #   and respond with a repeat act.
        if goal.patience > 1 and sys_act.act != 'repeat' and sys_act.act != 'badact' and \
                sys_act.act != 'null':
            if Settings.random.rand() < self.rand_decision_probs['Repeat']:
                self.agenda.push(DiaAct('repeat()'))
                return

        # Generate null action with small probability:
        #   user generates (silence or) something incomprehensible
        if Settings.random.rand() < self.rand_decision_probs['NullResp']:
            self.agenda.push(DiaAct('null()'))
            return

        if sys_act.act in self.receive_options and sys_act.act != 'null' and sys_act.act != 'badact':
            self.receive_options[sys_act.act](sys_act, goal)
        else:
            print 'Unknown act type in UMHdcSim.receive(): ' + sys_act.act
            self._receive_badact(goal)

    def respond(self, goal):
        """
        This method is called to get the user response.

        :param goal: of :class:`UMGoal`
        :type goal: instance
        :returns: (instance) of :class:`DiaActWithProb`
        """

        # If agenda is empty, push ByeAct on top.
        if self.agenda.size() == 0:
            self.agenda.push(DiaAct('bye()'))

        # Pop the top act off the agenda to form the user response.
        dap = self.agenda.pop()

        # If it created negate(name="!x") or deny(name="x", name="!x") or confirm(name="!x") just reqalts()
        for item in dap.items:
            if item.op == "!=":
                dap = DiaAct('reqalts()')
                break

        # Checking agenda for redundant constraints.
        self.agenda.filter_constraints(dap)

        if dap.act in ['thankyou', 'silence', 'repeat', 'ack', 'deny', 'confirm']:
            return self._normalise_act_no_rules(dap)

        if self.last_sys_act.act == 'reqmore':
            return self._normalise_act_no_rules(dap)

        # Ckecing whether we might remove the slot name for value dontcare in the planned act.
        if dap.act == 'inform' and not dap.items:
            print 'Error inform act with no slots is on agenda.'

        # In response to a request about a particular slot users often do not specify the slot
        # especially when the value is dontcare.
        if self.last_sys_act.act in ['request', 'confreq', 'select']:
            if dap.act == 'inform' and len(dap.items) > 0 and \
                    dap.items[0] is not None and dap.items[0].val == 'dontcare':
                if Settings.random.rand() < self.rand_decision_probs['NoSlotWithDontcare']:
                    dap.items[0].slot = None

        # Checking whether we might add a venue name ot the planned act.
        if dap.act == 'request' and len(dap.items) == 1:
            rec_ven = goal.requests['name']
            # If venue recommended, randomly decide to include the venue name in the request.
            if rec_ven is not None:
                if Settings.random.rand() < self.rand_decision_probs['AddVenueNameToRequest']:
                    dap.append('name', rec_ven)

        # Checking whether we might include additional constraints in the planned act.
        # When specifying a constraint, combine the act with additional constraints with some probability.
        if dap.act in ['inform', 'negate', 'hello', 'affirm']:
            inf_comb_count = 0
            while self.agenda.size() > 0 and \
                    (self.agenda.agenda_items[-1].act == 'inform' or
                     self.agenda.agenda_items[-1].act == 'request' and dap.act == 'hello'):
                if Settings.random.rand() < self.rand_decision_probs['InformCombination']:
                    inf_comb_count += 1
                    next_dap = self.agenda.pop()
                    for dip in next_dap.items:
                        dap.append(dip.slot, dip.val, dip.op == '!=')
                else:
                    break

        # Checking whether we might request a slot when specifying the type of venue.
        # When specifying the requestType constraint at the beginning of a dialogue,
        # occasionally request an additional requested slot
        if dap.act == 'request' and len(dap.items) > 0 and dap.items[0].slot in ['type', 'task', 'restaurant']:
            print 'Not completely implemented: RequestSlotAtStart'

        usr_output = self._normalise_act_no_rules(dap)
        self.last_user_act = usr_output
        return usr_output

    def _normalise_act_no_rules(self, dap):
        norm_act = copy.deepcopy(dap)
        norm_act.items = []

        for item in dap.items:
            keep_it = True
            val = item.val
            slot = item.slot

            if slot == 'task':
                keep_it = False
            elif dap.act == 'request' and val is None:
                if slot == 'name':
                    keep_it = False
                    if val is None:
                        norm_act.act = 'inform'
                elif slot == 'bar' or slot == 'restaurant' or slot == 'hotel':
                    norm_act.append('type', slot)
                    keep_it = False
            elif slot is None and val is not None and val != 'dontcare':
                keep_it = False
                norm_act.append('type', val)

            if keep_it:
                norm_act.append(slot, val)
        return norm_act

    def _receive_badact(self, goal):
        if goal.patience < 1:
            self.agenda.push(DiaAct('bye()'))
        else:
            self.agenda.push(copy.deepcopy(self.last_user_act))

    def _receive_hello(self, sys_act, goal):
        if not len(sys_act.items):
            if Settings.random.rand() < self.rand_decision_probs['Greeting']:
                self.agenda.push(DiaAct('hello()'))

    def _receive_bye(self, sys_act, goal):
        self.agenda.clear()
        self.agenda.push(DiaAct('bye()'))

    def _receive_reqmore(self, sys_act, goal):
        # Check if there are pending items on the agenda apart from bye().
        # If yes, just return and wait for them to be dealt with, or turn the top item of the agenda into an affirm.
        if self.agenda.size() > 1:
            next_dap = self.agenda.agenda_items[-1]
            if not next_dap.contains('type', goal.request_type):  # was hard coded to 'restaurant'
                if Settings.random.rand() < self.rand_decision_probs['CombAffirmWithAgdItem']:
                    # Responding with affirm and combine with next agenda item.
                    # Create an affirm act and combine it with the top item of the agenda if that specifies a constraint
                    # e.g. inform(type=bar) -> affirm(type=bar) or request(bar) -> affirm(=bar)
                    resp_act = DiaAct('affirm()')
                    # Combine this affirm with a constraint from the top of the agenda if possible.
                    if next_dap.act == 'inform' or (next_dap.act == 'request' and next_dap.contains_slot('type')):
                        for item in next_dap.items:
                            new_item = copy.copy(item)
                            if next_dap.act == 'request':
                                new_item.val = new_item.slot
                                new_item.slot = None
                            resp_act.append(new_item.slot, new_item.val, negate=(new_item.op == '!='))
                        self.agenda.pop()
                        self.agenda.push(resp_act)
                return

        # Check if there is an unsatisfied request on the goal
        if goal.are_all_requests_filled():
            # If all requests are filled,
            if Settings.random.rand() < self.rand_decision_probs['NoAfterReqmore']:
                # Occasionally just say no. A good policy can save turns by ending the conversation at this point.
                self.agenda.push(DiaAct('negate()'))
            else:
                informs = self.agenda.get_agenda_with_act('inform')
                confirms = self.agenda.get_agenda_with_act('confirm')
                # If there are still items on the agenda which need to be transmitted to the system,
                # then don't hangup!
                if len(informs) + len(confirms) > 0:
                    return
                # Say bye if there's nothing else to do.
                self.agenda.clear()
                self.agenda.push(DiaAct('bye()'))
        else:
            # If there are unsatisfied requests,
            if goal.is_venue_recommended():
                # If a venue has been recommended already, then ask for empty requests, like phone, address.
                unsatisfied = goal.get_unsatisfied_requests()
                for info in unsatisfied:
                    self.agenda.push(DiaAct('request(%s)' % info))

            # If no venue has been recommended yet, then asking reqmore() is pretty stupid.
            # Make the system loose a point by answering "yes!"
            else:
                if Settings.random.rand() < self.rand_decision_probs['YesAfterReqmore']:
                    # Nothing recommended yet, so just say yes.
                    self.agenda.push(DiaAct('affirm()'))
                else:
                    # Nothing recommended yet, so just ask for the request_type and all constraints.
                    usr_resp = DiaAct('inform(type=%s)' % goal.request_type)
                    for const in goal.constraints:
                        if const.val != 'dontcare':
                            slot = const.slot
                            value = goal.get_correct_const_value(slot)
                            if value is not None:
                                usr_resp.append(slot, value)
                            value = goal.get_correct_const_value(slot, negate=True)
                            if value is not None:
                                usr_resp.append(slot, value, negate=True)

                    self.agenda.push(usr_resp)

    def _receive_confirm(self, sys_act, goal):
        # Check the given information.
        if not self._receive_implicit_confirm(sys_act, goal):
            # The given info was not ok, so stop processing confirm act.
            return

        # Check explicit confirmation.
        if not self._receive_direct_implicit_confirm(sys_act, goal):
            # Item in system act needed correction: stop processing confirm act here.
            return

        # Given information is ok. Put an affirm on the agenda if next item on the agenda is an inform act,
        # can affirm and inform in one go.
        # affirm(), or affirm(a=x) if next agneda item is inform.
        new_affirm_act = DiaAct('affirm()')
        if self.agenda.size() > 0:
            top_item = self.agenda.agenda_items[-1]
            if top_item.act == 'inform':
                if Settings.random.rand() < self.rand_decision_probs['AffirmCombination']:
                    for item in top_item.items:
                        new_affirm_act.append(item.slot, item.val, negate=(item.op == '!='))
                    self.agenda.pop()
        self.agenda.push(new_affirm_act)

    def _receive_repeat(self, sys_act, goal):
        if self.last_user_act is not None and self.last_user_act.act != 'silence':
            self.agenda.push(copy.deepcopy(self.last_user_act))

    def _receive_select(self, sys_act, goal):
        # Receive select(slot=x, slot=y)
        slot = sys_act.items[0].slot
        value = sys_act.items[0].val
        if slot == 'name':
            self.agenda.push(DiaAct('inform(%s="%s")' % (slot, value)))
            # logger.error('select on name slot.')

        if not goal.contains_slot_const(slot):
            # If slot is not in the goal, get the correct value for it.
            print 'Slot %s in the given system act %s is not found in the user goal.' % (slot, str(sys_act))
            # random_val = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot)
            goal.add_const(slot, 'dontcare')
            self.agenda.push(DiaAct('inform(%s="%s")' % (slot, 'dontcare')))
        else:
            correct_val = goal.get_correct_const_value(slot)
            self.agenda.push(DiaAct('inform(%s="%s")' % (slot, correct_val)))
            return

    def _receive_confreq(self, sys_act, goal):
        """
        confreq(a=x,...,c=z,d): implicit confirm + request d.
        :param sys_act:
        :param goal:
        :return:
        """
        # Split into confirm(a=x,...c=z) and request(d)
        req_act = DiaAct('request()')
        for item in sys_act.items:
            if item.val is None or item.slot == 'option':
                req_act.append(item.slot, None)

        conf_act = DiaAct('confirm()')
        for item in sys_act.items:
            if item.val is not None and item.slot != 'option':
                conf_act.append(item.slot, item.val, negate=(item.op == '!='))

        # Process the implicit confirm() part. If it leads to any action then ignore the request() part.
        if self._receive_implicit_confirm(conf_act, goal):
            # Implicit confirmed items were ok.
            if self._receive_direct_implicit_confirm(conf_act, goal):
                # Implicit confirmed items were ok. Now process the request.
                self._receive_request(req_act, goal)

    def _receive_negate(self, sys_act, goal):
        self._receive_inform(sys_act, goal)

    def _receive_request(self, sys_act, goal):
        items = sys_act.items
        requested_slot = items[0].slot

        # Check if any options are given.
        if len(items) > 1:
            print 'request(a,b,...) is not supported: ' + sys_act

        '''
        First check if the system has actually already recommended the name of venue.
        If so, check if the user is still trying to get requested info.
        In that case, don't respond to the request (in at least some of the cases)
        but ask for the requested info.
        '''
        # Check if there is an unsatisfied request on the goal
        if goal.is_venue_recommended():
            for info in goal.requests:
                if goal.requests[info] is None:
                    self.agenda.push(DiaAct('request(name="%s",%s)' % (goal.requests['name'], info)))
                    return

        '''
        request(info)
        "Do you know the phone number of the place you are looking for?", etc.
        Just say no.
        '''
        if Ontology.global_ontology.is_user_requestable(slot=requested_slot):
            self.agenda.push(DiaAct('negate()'))
            return

        '''
        Handle valid requests
        '''
        answer_slots = [requested_slot]
        '''
        Case 1: Requested slot is somewhere on the agenda.
        '''
        # Go through the agenda and locate any corresponding inform() acts.
        # If you find one, move it to the top of the agenda.
        action_taken = False
        inform_items = self.agenda.get_agenda_with_act('inform')
        for agenda_act in inform_items:
            for item in agenda_act.items:
                if item.slot in answer_slots:
                    # Found corresponding inform() on agenda and moving it to top.
                    action_taken = True
                    self.agenda.remove(agenda_act)
                    self.agenda.push(agenda_act)
        if action_taken:
            return

        '''
        Case 2: Requested slot is not on the agenda, but there is another request() or inform() on the agenda.
        '''
        if not self.answer_req_always:
            if self.agenda.get_agenda_with_act('inform') != [] or self.agenda.get_agenda_with_act('request') != []:
                return

        '''
        Case 3: There is nothing on the agenda that would suit this request,
                but there is a corresponding constraint in the user goal.
        '''
        if goal.contains_slot_const(requested_slot):
            new_act = DiaAct('inform()')
            for slot in answer_slots:
                correct_val = goal.get_correct_const_value(slot)
                if correct_val is not None:
                    new_act.append(slot, correct_val)
                wrong_val = goal.get_correct_const_value(slot, negate=True)
                if wrong_val is not None:
                    new_act.append(slot, wrong_val, negate=True)
            self.agenda.push(new_act)
            return

        '''
        Case 4: There is nothing on the agenda or on the user goal.
        '''
        # Either repeat last user request or invent a value for the requested slot.
        f = Settings.random.rand()
        if f < self.rand_decision_probs['NewRequestResp1']:
            # Decided to randomly repeat one of the goal constraints.
            # Go through goal and randomly pick a request to repeat.
            random_val = goal.get_correct_const_value(
                requested_slot)  # copied here from below because random_val was not defined. IS THIS CORRECT?
            if len(goal.constraints) == 0:
                # No constraints on goal: say dontcare.
                self.agenda.push(DiaAct('inform(=dontcare)'))
                goal.add_const(slot=requested_slot, value=random_val)
                goal.add_prev_used(requested_slot, random_val)
            else:
                sampled_act = Settings.random.choice(goal.constraints)
                sampled_slot, sampled_op, sampled_value = sampled_act.slot, sampled_act.op, sampled_act.val
                self.agenda.push(DiaAct('inform(%s="%s")' % (sampled_slot, sampled_value)))

        elif f < self.rand_decision_probs['NewRequestResp1'] + self.rand_decision_probs['NewRequestResp2']:
            # Pick a constraint from the list of options and randomly invent a new constraint.
            # random_val = goal.getCorrectValueForAdditionalConstraint(requested_slot) # wrong method from dongho?
            random_val = goal.get_correct_const_value(requested_slot)
            if random_val is None:
                # requests about slots not in goal.
                if random_val is None:
                    # Again, none found. Setting it to dontcare.
                    random_val = 'dontcare'
                    self.agenda.push(DiaAct('inform(=%s)' % random_val))
                    goal.add_prev_used(requested_slot, random_val)
                else:
                    additional_slots = [requested_slot]
                    for slot in additional_slots:
                        rval = Ontology.global_ontology.getRandomValueForSlot(slot=slot)
                        self.agenda.push(DiaAct('inform(%s="%s")' % (slot, rval)))
            else:
                goal.add_const(slot=requested_slot, value=random_val)
                # goal.constraints[requested_slot] = random_val
                self.agenda.push(DiaAct('inform(%s="%s")' % (requested_slot, random_val)))
        else:
            # Decided to say dontcare.
            goal.add_const(slot=requested_slot, value='dontcare')
            goal.add_prev_used(requested_slot, 'dontcare')
            self.agenda.push(DiaAct('inform(%s="%s")' % (requested_slot, 'dontcare')))

    def _receive_affirm(self, sys_act, goal):
        self._receive_inform(sys_act, goal)

    def _receive_inform(self, sys_act, goal):
        # Check if the given inform act contains name=none.
        # If so, se the flag RELAX_CONSTRAINTS.
        possible_venue = []
        contains_name_none = False

        for item in sys_act.items:
            if item.slot == 'name':
                if item.op == '=' and item.val == 'none':
                    contains_name_none = True
                    self.relax_constraints = True
                elif item.op == '!=':
                    possible_venue.append(item.val)
                    contains_name_none = False
                    self.relax_constraints = False
                else:
                    self.relax_constraints = False

        # Reset requested slots right after the system recommend new venue.
        for item in sys_act.items:
            if item.slot == 'name' and item.op == '=' and item.val != 'none':
                if goal.requests['name'] != item.val:
                    goal.reset_requests()
                    break

        # Check the implicitly confirmed information.
        impl_confirm_ok = self._receive_implicit_confirm(sys_act, goal, False)
        if not impl_confirm_ok:
            return

        # If we get this far then all implicitly confirmed constraints were correctly understood.
        # If they don't match an item in the database, however, say bye or try again from beginning.
        sel_venue = None
        if self.use_new_goal_scenarios:
            change_goal = False

            if contains_name_none:
                change_goal = True
            elif self.first_venue_recommendation:
                self.first_venue_recommendation = False

                # Make a random choice of asking for alternatives,
                # even if the system has recommended another venue.
                f = Settings.random.rand()
                if f < self.rand_decision_probs['ReqAltsAfterVenRec1']:
                    # Ask for alternatives without changing the goal but add a !name in constraints.

                    # Insert name!=venue constraint.
                    goal.add_name_constraint(sys_act.get_value('name'), negate=True)

                    self.agenda.push(DiaAct('reqalts()'))
                    return

                elif f < self.rand_decision_probs['ReqAltsAfterVenRec1'] + \
                        self.rand_decision_probs['ReqAltsAfterVenRec2']:
                    # Do change the goal and ask for alternatives.
                    change_goal = True

                else:
                    # Decide not to ask for alternatives nor change the goal at this point.
                    goal.add_name_constraint(sys_act.get_value('name'))
            else:
                # After first venue recommendation we can't ask for alternatives again.
                goal.add_name_constraint(sys_act.get_value('name'))

            if change_goal:
                # Changing the goal.
                if len(goal.constraints) == 0:
                    change_goal = False
                else:
                    # Collect the constraints mentioned by the system act.
                    relax_candidates = []
                    for item in sys_act.items:
                        # Remember relax candidate that has to be set to dontcare.
                        set_dontcare = False
                        if contains_name_none and item.val == 'dontcare' and item.op == '!=':
                            set_dontcare = True
                        # Update candidate list
                        if item.slot not in ['name', 'type'] and \
                                Ontology.global_ontology.is_system_requestable(slot=item.slot) and \
                                item.val not in [None, goal.request_type] and \
                                (item.val != 'dontcare' or item.op == '!='):
                            relax_candidates.append((item.slot, set_dontcare))

                    # Pick a constraint to relax.
                    relax_dontcare = False
                    if len(relax_candidates) > 0:
                        index = Settings.random.randint(len(relax_candidates))
                        (relax_slot, relax_dontcare) = relax_candidates[index]
                    # Randomly pick a goal constraint to relax
                    else:
                        index = Settings.random.randint(len(goal.constraints))
                        relax_const = goal.constraints[index]
                        relax_slot = relax_const.slot

                    # Randomly decide whether to change it to another value or set it to 'dontcare'.
                    if relax_slot is not None:
                        # if type(relax_slot) not in [unicode, str]:
                        #    print relax_slot
                        #    logger.error('Invalid relax_slot type: %s in %s' % (type(relax_slot), relax_slot))
                        if goal.contains_slot_const('name'):
                            goal.remove_slot_const('name')

                        if Settings.random.rand() < self.rand_decision_probs['ConstraintRelax'] or relax_dontcare:
                            # Just set it to dontcare.
                            relax_value = 'dontcare'
                        elif relax_slot not in goal.prev_slot_values:
                            relax_value = 'dontcare'
                            goal.add_prev_used(relax_slot, relax_value)  # is this necessary?
                        else:
                            # Set it to a valid value for this slot that is different from the previous one.
                            relax_value = Ontology.global_ontology.getRandomValueForSlot(slot=relax_slot,
                                                                                         nodontcare=True,
                                                                                         notthese=goal.prev_slot_values[
                                                                                             relax_slot])
                            goal.add_prev_used(relax_slot, relax_value)

                        goal.replace_const(relax_slot, relax_value)

                        # Randomly decide whether to tell the system about the change or just request an alternative.
                        if not contains_name_none:
                            if Settings.random.rand() < self.rand_decision_probs['TellAboutChange']:
                                # Decide to tell the system about it.
                                self.agenda.push(DiaAct('reqalts(%s="%s")' % (relax_slot, relax_value)))
                            else:
                                # Decide not to tell the system about it.
                                # If the new goal constraint value is set to something other than dontcare,
                                # then add the slot to the list of requests, so that the user asks about it
                                # at some point in the dialogue.
                                # If it is set to dontcare, add name!=value into constraint set.
                                self.agenda.push(DiaAct('reqalts()'))
                                if relax_value == 'dontcare':
                                    goal.add_name_constraint(sys_act.get_value('name'), negate=True)
                                else:
                                    goal.requests[relax_slot] = None
                        else:
                            # After inform(name=none,...) always tell the system about the goal change.
                            self.agenda.push(DiaAct('reqalts(%s="%s")' % (relax_slot, relax_value)))
                        return

                    else:
                        # No constraint to relax.
                        change_goal = False

            else:  # change_goal == False
                # If name=none, ..., name!=x, ...
                if len(possible_venue) > 0:
                    # If # of possible venues is same to the number of name!=value constraints,
                    # that is, all possible venues are excluded by name!=value constraints.
                    # The user must relax them.
                    is_there_possible_venue = False
                    for venue in possible_venue:
                        if goal.is_satisfy_all_consts(DiaActItem('name', '=', venue)):
                            is_there_possible_venue = True

                    if not is_there_possible_venue:
                        goal.remove_slot_const('name')

                    # Remove possible venues violating name constraints.
                    copy_possible_venue = copy.copy(possible_venue)
                    for venue in copy_possible_venue:
                        if not goal.is_satisfy_all_consts(DiaActItem('name', '=', venue)):
                            possible_venue.remove(venue)

                    # 1) Choose venue from possible_venue, which satisfy the constraints.
                    sel_venue = Settings.random.choice(possible_venue)

                    # 2) Relax appropriate constraint from goal.
                    for cslot in copy.deepcopy(goal.constraints):
                        if not sys_act.contains_slot(cslot.slot):
                            # Constraint not found in system act: relax it.
                            goal.replace_const(cslot.slot, 'dontcare')

                            # Also remove any informs about this constraint from the agenda.
                            self.agenda.filter_acts_slot(cslot.slot)

        # Endif self.user_new_goal_scenarios == True
        if self.relax_constraints:
            # The given constraints were understood correctly but did not match a venue.
            if Settings.random.rand() < self.rand_decision_probs['ByeOrStartOver']:
                self.agenda.clear()
                self.agenda.push(DiaAct('bye()'))
            else:
                # self.agenda.push(DiaAct.DiaAct('inform(type=restaurant)'))
                self.agenda.push(DiaAct('inform(type=%s)' % goal.request_type))
            return

        '''
        If we get this far then all implicitly confirmed constraints are correct.
        Use the given information to fill goal request slots.
        '''
        for slot in goal.requests:
            if slot == 'name' and sel_venue is not None:
                goal.requests[slot] = sel_venue
            else:
                for item in sys_act.items:
                    if item.slot == slot:
                        if item.op != '!=':
                            goal.requests[slot] = item.val

        '''
        With some probability, change any remaining inform acts on the agenda to confirm acts.
        '''
        if Settings.random.rand() < self.rand_decision_probs['InformToConfirm']:
            for agenda_item in self.agenda.agenda_items:
                if agenda_item.act == 'inform':
                    if len(agenda_item.items) == 0:
                        print 'Empty inform act found on agenda.'
                    elif agenda_item.items[0].val != 'dontcare':
                        agenda_item.act = 'confirm'

        # Randomly decide to respond with thankyou() or ack(), or continue.
        if self.use_new_goal_scenarios:
            f = Settings.random.rand()
            if f < self.rand_decision_probs['ThankAck1']:
                self.agenda.push(DiaAct('thankyou()'))
                return
            elif f < self.rand_decision_probs['ThankAck1'] + self.rand_decision_probs['ThankAck2']:
                self.agenda.push(DiaAct('ack()'))
                return

        '''
        If empty slots remain in the goal, put a corresponding request for the first empty slot onto the agenda.
        If there are still pending acts on the agenda apart from bye(), though, then process those first,
        at least sometimes.
        '''
        if self.agenda.size() > 1:
            if Settings.random.rand() < self.rand_decision_probs['DealWithPending']:
                return

        # If empty goal slots remain, put a request on the agenda.
        if not goal.are_all_requests_filled():
            # Specify name in case the system giving complete list of venues matching constraints
            # inform(name=none, ..., name!=x, name!=y, ...)
            user_response = DiaAct('request()')
            if sel_venue is not None:
                # If user picked venue from multiple venues, it needs to specify selected name in request act.
                # If only one possible venue was offered, it specifies the name with some probability.
                user_response.append('name', sel_venue)

            one_added = False
            for slot in goal.requests:
                value = goal.requests[slot]
                if value is None:
                    if not one_added:
                        user_response.append(slot, None)
                        one_added = True
                    else:
                        # Add another request with some probability
                        if Settings.random.rand() < self.rand_decision_probs['AddSlotToReq']:
                            user_response.append(slot, None)
                        else:
                            break

            self.agenda.push(user_response)

    def _receive_direct_implicit_confirm(self, sys_act, goal):
        """
        Deals with implicitly confirmed items that are not on the user goal.
        These are okay in system inform(), but not in system confirm() or confreq().
        In this case, the user should mention that slot=dontcare.
        :param sys_act:
        :param goal:
        :return:
        """
        for item in sys_act.items:
            slot = item.slot
            val = item.val
            if slot in ['count', 'option', 'type']:
                continue
            if not goal.contains_slot_const(slot) and val != 'dontcare':
                self.agenda.push(DiaAct('negate(%s="dontcare")' % slot))
                return False

        # Explicit confirmations okay.
        return True

    def _receive_implicit_confirm(self, sys_act, goal, fromconfirm=True):
        """
        This method is used for checking implicitly confirmed items.
        :param sys_act:
        :param goal:
        :return: True if all the items are consistent with the user goal.
                 If there is a mismatch, then appropriate items are added to the agenda and
                 the method returns False.
        """
        contains_name_none = False
        contains_count = False
        is_inform_requested = False

        # First store all possible values for each unique slot in the list of items and check for name=none
        slot_values = {}
        informable_s = Ontology.global_ontology.get_informable_slots()
        requestable_s = Ontology.global_ontology.get_requestable_slots()
        for item in sys_act.items:
            if item.slot != 'count' and item.slot != 'option':
                if item.slot not in slot_values:
                    slot_values[item.slot] = set()
                slot_values[item.slot].add(item)
            elif item.slot == 'count':
                contains_count = True
                count = item.val
            elif item.slot == 'option':
                print 'option in dialog act is not supported.'

            if item.slot == 'name' and item.val == 'none':
                contains_name_none = True

            if item.slot in requestable_s and item.slot not in informable_s:
                is_inform_requested = True

        # Check if all the implicitly given information is correct. Otherwise reply with negate or deny.
        do_exp_confirm = False
        for item in sys_act.items:
            correct_val = None
            correct_slot = None
            do_correct_misunderstanding = False
            # negation = (item.op == '!=')

            if item.slot in ['count', 'option']:
                continue

            # Exclude slots that are info keys in the ontology, go straight to the next item.
            if Ontology.global_ontology.is_user_requestable(slot=item.slot) and \
                    not goal.contains_slot_const(item.slot):
                continue

            # If an implicitly confirmed slot is not present in the goal,
            # it doesn't really matter, unless:
            # a) the system claims that there is no matching venue, or
            # b) the system explicitly confirmed this slot.
            if not goal.contains_slot_const(item.slot):
                pass
            else:
                correct_val = goal.get_correct_const_value(item.slot)
                wrong_val = goal.get_correct_const_value(item.slot, negate=True)

                # Current negation: if the goal is slot!=value
                if correct_val is not None and wrong_val is not None and correct_val == wrong_val:
                    print str(goal), 'Not possible'

                # Conflict between slot!=value on user goal and slot=value in system act.
                if wrong_val is not None and not goal.is_satisfy_all_consts(slot_values[item.slot]):
                    if contains_name_none:
                        if item.slot == 'name':
                            # Relax constraint for !name because of name=none.
                            goal.remove_slot_const('name', negate=True)
                        else:
                            # Must correct it because of name=none.
                            do_correct_misunderstanding = True
                    # System informed wrong venue.
                    else:
                        do_correct_misunderstanding = True

                if item.slot == 'name' and not is_inform_requested and not do_correct_misunderstanding:
                    continue

                # Conflict between slot=value on user goal, and slot=other or slot!=other in system act
                if correct_val is not None and not goal.is_satisfy_all_consts(slot_values[item.slot]):
                    # If the system act contains name=none, then correct the misunderstanding in any case.
                    if contains_name_none:
                        if item.slot != 'name':
                            do_correct_misunderstanding = True
                    # If it doesn't, then only correct the misunderstanding if the user goal constraints say so
                    elif correct_val != 'dontcare':
                        do_correct_misunderstanding = True

            # If all constraints mentioned by user but some are not implicitly confirmed in system act,
            # confirm one such constraint with some probability
            planned_response_act = None
            if not contains_name_none:
                # Now find constraints not mentioned in system act, if any
                confirmable_consts = []
                for const in goal.constraints:
                    slots_to_confirm = [const.slot]
                    for sit in slots_to_confirm:
                        s1 = sit

                        # Check if the system act contains the slot value pair (const_slot and its value)
                        found = False
                        v1 = goal.get_correct_const_value(s1)
                        if sys_act.contains(s1, v1):
                            found = True
                        v1_neg = goal.get_correct_const_value(s1, negate=True)
                        if v1_neg is not None and sys_act.contains_slot(s1) and sys_act.get_value(s1) != v1_neg:
                            found = True

                        if not found and const.val not in ['dontcare', 'none', '**NONE**']:
                            confirmable_consts.append(const)

                # Now pick a constraint to confirm
                if len(confirmable_consts) > 0:
                    rci = Settings.random.choice(confirmable_consts)
                    planned_response_act = DiaAct('confirm()')
                    planned_response_act.append(rci.slot, rci.val)

            if do_correct_misunderstanding:
                # Depending on the patience level, say bye with some probability (quadratic function of patience level)
                if not self.patience_old_style:
                    prob1 = float(goal.patience ** 2) / self.max_patience ** 2
                    prob2 = float(2 * goal.patience) / self.max_patience
                    prob = prob1 - prob2 + 1
                    if Settings.random.rand() < prob:
                        # Randomly decided to give up
                        self.agenda.clear()
                        self.agenda.push(DiaAct('bye()'))
                        return False

                # Pushing negate or deny onto agenda to correct misunderstanding.
                # Make a random decision as to whether say negate(a=y) or deny(a=y,a=z), or
                # confirm a constraint not mentioned in system act.
                # If the type is wrong, say request(whatever).

                if item.slot != 'type':
                    correct_it = False
                    if planned_response_act is None:
                        correct_it = True
                    else:
                        if Settings.random.rand() >= self.rand_decision_probs['OverruleCorrection']:
                            # Decided to correct the system.
                            correct_it = True

                    if correct_it:
                        if correct_val is None:
                            correct_val = 'dontcare'

                        cslot = item.slot
                        if correct_slot is not None:
                            cslot = correct_slot

                        planned_response_act = None
                        if wrong_val is not None:
                            planned_response_act = DiaAct('inform(%s!="%s")' % (cslot, wrong_val))
                            # planned_response_act = DiaAct.DiaAct('negate(%s="%s")' % (cslot, correct_val))
                        else:
                            f = Settings.random.rand()
                            if f < self.rand_decision_probs['CorrectingAct1']:
                                planned_response_act = DiaAct('negate(%s="%s")' % (cslot, correct_val))
                            elif f < self.rand_decision_probs['CorrectingAct1'] + \
                                    self.rand_decision_probs['CorrectingAct2']:
                                planned_response_act = DiaAct('deny(%s="%s",%s="%s")' % (item.slot, item.val,
                                                                                         cslot, correct_val))
                            else:
                                planned_response_act = DiaAct('inform(%s="%s")' % (cslot, correct_val))
                else:
                    planned_response_act = DiaAct('inform(type=%s)' % goal.request_type)

                self.agenda.push(planned_response_act)

                # Resetting goal request slots.
                goal.reset_requests()
                return False

            # The system's understanding is correct so far, but with some changes,
            # the user decide to confirm a random constraints.
            elif planned_response_act is not None and not do_exp_confirm and not fromconfirm:
                if Settings.random.rand() < self.rand_decision_probs['ConfirmRandomConstr']:
                    # Decided to confirm a random constraint.
                    self.agenda.push(planned_response_act)
                    do_exp_confirm = True

            elif contains_name_none:
                # No correction required in case of name=none: set goal status systemHasInformedNameNone=True
                goal.system_has_informed_name_none = True

        # The user decide to confirm a random constraints.
        if do_exp_confirm:
            goal.fill_requests(sys_act.items)
            return False

        # Implicit confirmations okay.
        return True
