from DRLPolicy.utils import Settings


class EvaluationManager(object):

    def __init__(self):
        self.evaluator = None
        self.final_reward = None

        self._load_evaluator()

    def restart(self):
        if self.evaluator is not None:
            self.evaluator.restart()
        self.final_reward = None

    def get_turn_reward(self, turn_info):
        if self.evaluator is None:
            self._load_evaluator()
        return self.evaluator.get_turn_reward(turn_info)

    def get_final_reward(self, final_info=None):
        if self.evaluator is not None:
            return self.evaluator.get_final_reward(final_info)
        return 0

    def do_training(self):
        if self.evaluator is not None:
            do_training = self.evaluator.do_training()
        else:
            do_training = True     # by default all dialogues are potentially used for training
        return do_training

    def print_summary(self):
        """
        Prints the history over all dialogs run thru simulate.
        """
        if self.evaluator is not None:
            self.evaluator.print_summary()

    def _load_evaluator(self):
        evaluator = 'objective'

        if Settings.config.has_option('eval', 'successmeasure'):
            evaluator = Settings.config.get('eval', 'successmeasure')

        if evaluator == "objective":
            from SuccessEvaluator import ObjectiveSuccessEvaluator
            self.evaluator = ObjectiveSuccessEvaluator()
            self.evaluator.restart()
