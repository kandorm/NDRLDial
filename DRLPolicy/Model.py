import sys

sys.path.insert(0, '.')
from DRLPolicy.utils import Settings
from DRLPolicy.ontology import Ontology
from DRLPolicy.Simulate import SimulationSystem
from DRLPolicy.policy.PolicyManager import PolicyManager
from DRLPolicy.utils.commandparser import DRLPolicyOptParser

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


class DRLPolicy(object):

    def __init__(self, config=None):
        # not enough info to execute
        if config is None:
            print "Please specify config file ..."
            return

        Settings.init(config)
        Ontology.init_global_ontology()

        self.max_epoch = 1
        self.batches_per_epoch = 5
        self.train_batch_size = 100
        self.eval_batch_size = 100
        self.eval_per_batch = False
        self.policy_type = ''

        if Settings.config.has_option('train', 'max_epoch'):
            self.max_epoch = Settings.config.getint('train', 'max_epoch')
        if Settings.config.has_option('train', 'batches_per_epoch'):
            self.batches_per_epoch = Settings.config.getint('train', 'batches_per_epoch')
        if Settings.config.has_option('train', 'train_batch_size'):
            self.train_batch_size = Settings.config.getint('train', 'train_batch_size')
        if Settings.config.has_option('train', 'eval_batch_size'):
            self.eval_batch_size = Settings.config.getint('train', 'eval_batch_size')
        if Settings.config.has_option('train', 'eval_per_batch'):
            self.eval_per_batch = Settings.config.getboolean('train', 'eval_per_batch')
        if Settings.config.has_option('policy', 'policytype'):
            self.policy_type = Settings.config.get('policy', 'policytype')

        self.policy_manager = PolicyManager()

    def train_policy(self):
        # Just for training
        from KB import KnowledgeBase
        KnowledgeBase.init_global_kb()

        epoch = 0
        max_epoch = self.max_epoch
        while epoch < max_epoch:
            epoch += 1
            for batch_id in range(self.batches_per_epoch):
                print '==== Training iteration=', batch_id, 'num-dialogs=', self.train_batch_size
                self.train_batch()
                if self.eval_per_batch:
                    self.eval_policy()

            self.eval_policy()

    def act_on(self, belief_state, entities):
        return self.policy_manager.act_on(belief_state,entities)

    def eval_policy(self):
        Settings.config.set("policy", "learning", 'False')
        if self.policy_type == "gp":
            Settings.config.set("gpsarsa", "scale", "1")

        simulator = SimulationSystem()
        simulator.run_dialogs(self.eval_batch_size)

    def train_batch(self):
        Settings.config.set("policy", "learning", 'True')
        if self.policy_type == "gp":
            Settings.config.set("gpsarsa", "scale", "3")

        simulator = SimulationSystem()
        simulator.run_dialogs(self.train_batch_size)


if __name__ == '__main__':

    args = DRLPolicyOptParser()
    config = args.config

    model = DRLPolicy(config)
    if args.mode == 'train':
        model.train_policy()
