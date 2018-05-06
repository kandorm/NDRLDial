from DRLPolicy.usersimulator.SimulatedUser import SimulatedUser
from DRLPolicy.DialogueAgent import DialogueAgent


class SimulationSystem(object):

    def __init__(self):
        self.agent = DialogueAgent()
        self.simulator = SimulatedUser()

    def run_dialogs(self, num_dialogs):
        """
        run a loop over the run() method for the given number of dialogues.
        :param num_dialogs: number of dialogues to loop over.
        :type num_dialogs: int
        :return: None
        """
        for idx in range(num_dialogs):
            self.run()
        self.agent.power_down()

    def run(self, debug=False):
        # reset the user model
        self.simulator.restart()

        if debug:
            print self.simulator.um.goal

        ending_dialogue = False
        # initialize agent, sys_act here is usually 'hello()'
        sys_act = self.agent.start_call(self.simulator)

        while not ending_dialogue:

            # USER ACT:
            # ---------------------------------------------------------------------------------------------------------
            # last system action type: str
            sys_act = self.agent.retrieve_last_sys_act()
            # user action type:~utils.dact.DiaAct
            user_act = self.simulator.act_on(sys_act)

            if debug:
                print sys_act
                print user_act

            # SYSTEM ACT:
            # ---------------------------------------------------------------------------------------------------------
            sys_act = self.agent.continue_call(user_act, self.simulator)

            if 'bye' == user_act.act or 'bye' == sys_act.act:
                ending_dialogue = True

        self.agent.end_call(self.simulator)
