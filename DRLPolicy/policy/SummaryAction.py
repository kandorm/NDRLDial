from DRLPolicy.policy import SummaryUtils
from DRLPolicy.utils import Settings
from DRLPolicy.ontology import Ontology


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
        self.inform_count_accepted = 3

        if Settings.config.has_option("summaryacts", "informmask"):
            self.inform_mask = Settings.config.getboolean('summaryacts', 'informmask')
        if Settings.config.has_option("summaryacts", "requestmask"):
            self.request_mask = Settings.config.getboolean('summaryacts', 'requestmask')
        if Settings.config.has_option("summaryacts", "informcountaccepted"):
            self.inform_count_accepted = Settings.config.getint('summaryacts', 'informcountaccepted')

        if not empty:
            slots = Ontology.global_ontology.get_system_requestable_slots()
            for slot in slots:
                self.action_names.append("request_" + slot)
                self.action_names.append("confirm_" + slot)
                if confreq:
                    for slot2 in slots:
                        self.action_names.append("confreq_" + slot + "_" + slot2)

            self.action_names += ["inform",
                                  "inform_byname"]

    # MASK OVER SUMMARY ACTION SET
    # ------------------------------------------------------------------------------------

    def get_none_executable(self, belief_state):
        """
        Set of rules defining the mask over the action set, given the current belief state
        :param belief_state: the state the policy acts on
        :type belief_state: dict
        :return: list of non-executable (masked) actions
        """
        nonexec = []

        array_slot_summary = SummaryUtils.arraySlotSummary(belief_state)

        for action in self.action_names:
            mask_action = False

            if action == "inform":
                acceptance_list = SummaryUtils.getTopBeliefs(belief_state)
                if len(acceptance_list) < self.inform_count_accepted:
                    mask_action = True
                if mask_action and self.inform_mask:
                    nonexec.append(action)

            elif action == "inform_byname":
                if SummaryUtils.getTopBelief(belief_state['name'])[0] == 'none':
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

        return nonexec

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
            output = self.getInformByConstraints(belief_state, entities)
        elif action == "inform_byname":
            output = self.getInformByName(belief_state, entities)
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

    def getInformByConstraints(self, belief_state, entities):
        accepted_values = SummaryUtils.getTopBeliefs(belief_state)
        constraints = SummaryUtils.getConstraints(accepted_values)
        return SummaryUtils.getInformByConstraints(constraints, entities)

    def getInformByName(self, belief_state, entities):
        requested_slots = SummaryUtils.getRequestedSlots(belief_state)
        name = SummaryUtils.getTopBelief(belief_state['name'])[0]
        return SummaryUtils.getInformRequestedSlots(requested_slots, name, entities)
