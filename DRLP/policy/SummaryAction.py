import sys
from DRLP.policy import SummaryUtils
from DRLP.utils import Settings
from DRLP.ontology import Ontology


class SummaryAction(object):
    """
    The summary action class encapsulates the functionality of
    a summary action along with the conversion from summary to master actions.
    .. Note::
        The list of all possible summary actions are defined in this class.
    """
    def __init__(self, empty=False, confreq=False):

        self.action_names = []
        self._array_slot_summary = None

        self.inform_mask = True
        self.request_mask = True
        self.inform_count_accepted = 1
        self.has_control = False
        self.has_request = True
        self.has_confirm = True

        if Settings.config.has_option("summaryacts", "informmask"):
            self.inform_mask = Settings.config.getboolean('summaryacts', 'informmask')
        if Settings.config.has_option("summaryacts", "requestmask"):
            self.request_mask = Settings.config.getboolean('summaryacts', 'requestmask')
        if Settings.config.has_option("summaryacts", "informcountaccepted"):
            self.inform_count_accepted = Settings.config.getint('summaryacts', 'informcountaccepted')
        if Settings.config.has_option("summaryacts", "has_control"):
            self.has_control = Settings.config.getboolean('summaryacts', 'has_control')
        if Settings.config.has_option("summaryacts", "has_request"):
            self.has_request = Settings.config.getboolean('summaryacts', 'has_request')
        if Settings.config.has_option("summaryacts", "has_confirm"):
            self.has_confirm = Settings.config.getboolean('summaryacts', 'has_confirm')

        if not empty:
            slots = Ontology.global_ontology.get_system_requestable_slots()
            for slot in slots:
                self.action_names.append("request_" + slot)
                self.action_names.append("confirm_" + slot)
                if confreq:
                    for slot2 in slots:
                        self.action_names.append("confreq_" + slot + "_" + slot2)

            self.action_names += ["inform",
                                  'inform_alternatives']

    # MASK OVER SUMMARY ACTION SET
    # ------------------------------------------------------------------------------------

    def get_none_executable(self, belief_state, entities):
        """
        Set of rules defining the mask over the action set, given the current belief state
        :param belief_state: the state the policy acts on
        :type belief_state: dict
        :return: list of non-executable (masked) actions
        """
        nonexec = []

        if not self.has_request:
            slots = Ontology.global_ontology.get_system_requestable_slots()
            for slot in slots:
                nonexec.append("request_" + slot)

        if not self.has_confirm:
            slots = Ontology.global_ontology.get_system_requestable_slots()
            for slot in slots:
                nonexec.append("confirm_" + slot)

        if not self.has_control:
            return nonexec

        array_slot_summary = SummaryUtils.arraySlotSummary(belief_state)

        for action in self.action_names:
            mask_action = False

            if action == "inform":
                count_accepted = len(SummaryUtils.getTopBeliefs(belief_state))
                if count_accepted < self.inform_count_accepted:
                    mask_action = True
                if mask_action and self.inform_mask:
                    nonexec.append(action)

            elif action == 'inform_alternatives':
                if len(belief_state['name']) < 1:
                    mask_action = True
                if mask_action and self.inform_mask:
                    nonexec.append(action)

            elif "request_" in action:
                pass

            elif "confirm_" in action:
                slot_summary = array_slot_summary[action.split("_")[1]]
                top_prob = slot_summary['TOPHYPS'][0][1]
                if top_prob == 0:
                    mask_action = True
                if mask_action and self.request_mask:
                    nonexec.append(action)

            elif "confreq_" in action:
                slot_summary = array_slot_summary[action.split("_")[1]]
                top_prob = slot_summary['TOPHYPS'][0][1]
                if top_prob == 0:
                    mask_action = True
                if mask_action and self.request_mask:
                    nonexec.append(action)

        nonexec = list(set(nonexec))

        return nonexec

    def get_executable_mask(self, belief_state, entities):

        exec_mask = []
        non_exec = self.get_none_executable(belief_state, entities)
        for action in self.action_names:
            if action in non_exec:
                exec_mask.append(-sys.maxint)
            else:
                exec_mask.append(0.0)

        return exec_mask

    def convert(self, belief_state, action, entities):
        """
        Converts the given summary action into a master action based on the current belief and the last system action.

        :param belief_state: (dict) the current master belief
        :param action: (string) the summary action to be converted to master action
        :param entities: search results from knowledge base (match to the constraints, if no constraints: random 10)
        """

        self._array_slot_summary = SummaryUtils.arraySlotSummary(belief_state)

        if "request_" in action:
            output = self.getRequest(action.split("_")[1])
        elif "confirm_" in action:
            output = self.getConfirm(action.split("_")[1])
        elif "confreq_" in action:
            output = self.getConfReq(action.split("_")[1], action.split("_")[2])
        elif action == "inform":
            output = self.getInform(belief_state, entities)
        elif action == "inform_alternatives":
            output = self.getInformAlternatives(belief_state, entities)
        else:
            output = ""
        return output

    # CONVERTING METHODS FOR EACH SPECIFIC ACT:
    # ------------------------------------------------------------------------------------
    def getRequest(self, slot):
        return 'request({})'.format(slot)

    def getConfirm(self, slot):
        summary = self._array_slot_summary[slot]
        top_value = summary['TOPHYPS'][0][0]
        return 'confirm({}="{}")'.format(slot, top_value)

    def getConfReq(self, cslot, rslot):
        summary = self._array_slot_summary[cslot]
        top_value = summary['TOPHYPS'][0][0]
        return 'confreq({}="{}",{})'.format(cslot, top_value, rslot)

    def getInform(self, belief_state, entities):
        requested_slots = SummaryUtils.getRequestedSlots(belief_state)
        accepted_values = SummaryUtils.getTopBeliefs(belief_state)
        constraints = SummaryUtils.getConstraints(accepted_values)
        if len(requested_slots):
            if len(belief_state['name']):
                name = belief_state['name'][-1]
            else:
                name = None
            return SummaryUtils.getInformRequestedSlots(requested_slots, name, constraints, entities)

        return SummaryUtils.getInformByConstraints(constraints, entities)

    def getInformAlternatives(self, belief_state, entities):
        recommended_list = belief_state['name']
        accepted_values = SummaryUtils.getTopBeliefs(belief_state)
        return SummaryUtils.getInformAlternativeEntities(accepted_values, recommended_list, entities)
