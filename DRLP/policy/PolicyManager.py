from DRLP.utils import Settings


class PolicyManager(object):

    def __init__(self):
        self.policy = None
        self._load_policy()

    def restart(self):
        self.policy.restart()

    def act_on(self, belief_state, entities):
        if self.policy is None:
            self._load_policy()
        sys_act = self.policy.act_on(belief_state, entities)
        return sys_act

    def record(self, reward):
        """
        Records the current turn reward.
        This method is called each turn by the :class:`~DialogueAgent.DialogueAgent`.

        :param reward: the turn reward to be recorded
        :type reward: int
        :returns: None
        """
        if self.policy is not None:
            self.policy.record(reward)

    def finalize_record(self, final_reward):
        """
        Records the final rewards.

        This method is called once at the end of each dialogue by the :class:`~DRLP.DialogueAgent`.

        :param final_reward: final reward
        :type final_reward: int
        :returns: None
        """
        if self.policy is not None:
            self.policy.finalize_record(final_reward)

    def train(self, do_training):
        if do_training:
            self.policy.train()

    def save_policy(self):
        if self.policy is not None:
            self.policy.save_policy()

    def _load_policy(self):

        policy_type = 'gp'
        in_policy_file = ''
        out_policy_file = ''
        learning = False

        if Settings.config.has_option('policy', 'policytype'):
            policy_type = Settings.config.get('policy', 'policytype')
        if Settings.config.has_option('policy', 'learning'):
            learning = Settings.config.getboolean('policy', 'learning')
        if Settings.config.has_option('policy', 'inpolicyfile'):
            in_policy_file = Settings.config.get('policy', 'inpolicyfile')
        if Settings.config.has_option('policy', 'outpolicyfile'):
            out_policy_file = Settings.config.get('policy', 'outpolicyfile')

        if policy_type == 'gp':
            from DRLP.policy.gp.GPPolicy import GPPolicy
            self.policy = GPPolicy(learning, in_policy_file, out_policy_file)
            self.policy.restart()
        elif policy_type == 'a2c':
            from DRLP.policy.a2c.A2CPolicy import A2CPolicy
            self.policy = A2CPolicy(learning, in_policy_file, out_policy_file)
            self.policy.restart()

    def get_last_system_action(self):
        if self.policy is not None:
            return self.policy.last_system_action
        return None
