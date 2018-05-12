from DRLP.policy.SummaryAction import SummaryAction
from DRLP.utils.dact import DiaAct
from DRLP.utils import Settings


class Policy(object):

    def __init__(self, learning=False, empty=False):
        self.prev_belief = None
        self.act_to_be_recorded = None
        self.last_system_action = None  # accessed from outside of policy

        self.learning = learning
        self.start_with_hello = True
        self.useconfreq = False

        if Settings.config.has_option('policy', 'startwithhello'):
            self.start_with_hello = Settings.config.getboolean('policy', 'startwithhello')
        if Settings.config.has_option('policy', 'useconfreq'):
            self.useconfreq = Settings.config.getboolean('policy', 'useconfreq')

        self.actions = SummaryAction(empty, self.useconfreq)
        # Total number of system actions.
        self.num_actions = len(self.actions.action_names)

        self.episode = Episode()

    def act_on(self, belief_state, entities):
        """
        Main policy method: mapping of belief state to system action.

        This method is automatically invoked by the agent at each turn after tracking the belief state.

        May initially return 'hello()' as hardcoded action. Keeps track of last system action and last belief state.

        :param belief_state: the belief state to act on
        :type belief_state: dict
        :param entities: search results from knowledge base (match to the constraints, if no constraints: random 10)
        :type entities: list
        :returns: the next system action of type :class:`~utils.dact.DiaAct`
        """
        if self.last_system_action is None and self.start_with_hello:
            sys_act = 'hello()'
        else:
            sys_act = self.next_action(belief_state, entities)
        self.last_system_action = sys_act
        self.prev_belief = belief_state
        sys_action = DiaAct(sys_act)
        return sys_action

    def record(self, reward, weight=None, state=None, action=None):
        """
        Records the current turn reward.
        This method is automatically executed by the agent at the end of each turn.

        :param reward: the turn reward to be recorded
        :param weight:
        :param state:
        :param action:
        :type reward: int
        """
        if self.episode is None:
            self.episode = Episode()
        if self.act_to_be_recorded is None:
            self.act_to_be_recorded = self.last_system_action

        if state is None:
            state = self.prev_belief
        if action is None:
            action = self.act_to_be_recorded

        c_state, c_action = self.convert_state_action(state, action)

        if weight is None:
            self.episode.record(state=c_state, action=c_action, reward=reward)
        else:
            self.episode.record(state=c_state, action=c_action, reward=reward, weight=weight)

        self.act_to_be_recorded = None

    def finalize_record(self, reward):
        """
        Records the final reward along with the terminal system action and terminal state.
        To change the type of state/action override :func:`~convertStateAction`.

        This method is automatically executed by the agent at the end of each dialogue.

        :param reward: the final reward
        :type reward: int
        :returns: None
        """
        if self.episode is None:
            self.episode = Episode()
        terminal_state, terminal_action = self.convert_state_action(TerminalState(), TerminalAction())
        self.episode.record(state=terminal_state, action=terminal_action, reward=reward)
        return

    def convert_state_action(self, state, action):
        return State(state), Action(action)

    #########################################################
    # interface methods
    #########################################################
    def next_action(self, belief_state, entities):
        """
        Interface method for selecting the next system action. Should be overridden by sub-class.

        This method is automatically executed by :func:`~act_on` thus at each turn.

        :param belief_state: the state the policy acts on
        :type belief_state: dict
        :param entities: search results from knowledge base (match to the constraints, if no constraints: random 10)
        :type entities: list
        :returns: the next system action
        """
        pass

    def train(self):
        """
        Interface method for initiating the training. Should be overridden by sub-class.

        This method is automatically executed by the agent at the end of each dialogue if learning is True.

        This method is called at the end of each dialogue by :class:`~policy.PolicyManager.PolicyManager`
        if learning is enabled for the given domain policy.
        """
        pass

    def save_policy(self):
        """
        Saves the learned policy model to file. Should be overridden by sub-class.

        This method is automatically executed by the agent either at certain intervals or
        at least before shutting down the agent.

        """
        pass

    def restart(self):
        """
        Restarts the policy. Resets internal variables.

        This method is automatically executed by the agent at the end/beginning of each dialogue.
        """
        self.last_system_action = None
        self.prev_belief = None
        self.act_to_be_recorded = None

        self.episode = Episode()


class State(object):
    """
    Dummy class representing one state. Used for recording and may be overridden by sub-class.
    """
    def __init__(self, state):
        self.state = state


class Action(object):
    """
    Dummy class representing one action. Used for recording and may be overridden by sub-class.
    """
    def __init__(self, action):
        self.act = action


class TerminalState(object):
    """
    Dummy class representing one terminal state. Used for recording and may be overridden by sub-class.
    """
    def __init__(self):
        self.state = "TerminalState"


class TerminalAction(object):
    """
    Dummy class representing one terminal action. Used for recording and may be overridden by sub-class.
    """
    def __init__(self):
        self.act = "TerminalAction"


class Episode(object):
    """
    An episode encapsulates the state-action-reward triplet which may be used for learning.
    Every entry represents one turn.
    The last entry should contain :class:`~TerminalState` and :class:`~TerminalAction`
    """
    def __init__(self):
        self.total_reward = 0
        self.total_weight = 0
        self.state_trace = []
        self.action_trace = []
        self.reward_trace = []

    def record(self, state, action, reward, weight=None):
        self.total_reward += reward
        if weight is not None:
            self.total_weight += weight
        self.state_trace.append(state)
        self.action_trace.append(action)
        self.reward_trace.append(reward)

    def check(self):
        """
        Checks whether length of internal state action and reward lists are equal.
        """
        assert(len(self.state_trace) == len(self.action_trace))
        assert(len(self.state_trace) == len(self.reward_trace))

    def get_weighted_reward(self):
        """
        Returns the reward weighted by normalised accumulated weights. Used for multiagent learning in committee.

        :returns: the reward weighted by normalised accumulated weights
        """
        reward = self.total_reward
        if self.total_weight != 0:
            # we subtract 1 as the last entry is TerminalState
            norm_weight = self.total_weight / (len(self.state_trace) - 1)
            reward *= norm_weight
        return reward
