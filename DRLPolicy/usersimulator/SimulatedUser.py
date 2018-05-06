from DRLPolicy.utils.dact import DiaAct
from DRLPolicy.usersimulator.UserModel import UserModel


class SimulatedUser(object):

    def __init__(self):
        self.um = UserModel()

    def restart(self):
        """
        Resets all components (**User Model**) that are statefull.
        :returns: None
        """
        self.um.init()

    def act_on(self, sys_act_string):
        """
        Through the UserModel member, receives the system action and then responds.
        :param sys_act_string: system action
        :type sys_act_string: unicode str
        :returns: (str) user action
        """
        sys_act = DiaAct(sys_act_string)
        self.um.receive(sys_act)
        user_act = self.um.respond()

        return user_act
