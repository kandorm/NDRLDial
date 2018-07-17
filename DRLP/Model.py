import sys

sys.path.insert(0, '.')
from DRLP.utils import Settings
from DRLP.ontology import Ontology
from DRLP.Simulate import SimulationSystem
from DRLP.policy.PolicyManager import PolicyManager
from DRLP.utils.commandparser import DRLPOptParser

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


class DRLP(object):

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

        self.policy_manager = None

    def train_policy(self):
        # Just for training
        from KB import KnowledgeBase
        KnowledgeBase.init_global_kb()

        epoch = 0
        max_epoch = self.max_epoch
        while epoch < max_epoch:
            epoch += 1

            if epoch == 1:
                random_seed = Settings.config.getint("general", "seed")
            else:
                random_seed = Settings.set_seed(None)
                Settings.config.set("general", "seed", str(random_seed))
            print '\n========================================================================'
            print '========= Epoch', epoch, '/', max_epoch, 'Seed:', random_seed
            print '========================================================================\n'

            for batch_id in range(self.batches_per_epoch):
                print '\n========= Training iteration=', batch_id + 1, '/', self.batches_per_epoch, 'num-dialogs=', \
                    self.train_batch_size, '=========\n'
                self.train_batch()
                if self.eval_per_batch:
                    self.eval_policy()

            self.eval_policy()

    def act_on(self, belief_state, entities):
        Settings.config.set("policy", "learning", 'False')
        Settings.config.set("policy", "startwithhello", "False")
        Settings.config.set("summaryacts", "has_control", "False")
        if self.policy_type == "gp":
            Settings.config.set("gpsarsa", "scale", "1")

        if not self.policy_manager:
            self.policy_manager = PolicyManager()

        return self.policy_manager.act_on(belief_state, entities)

    def eval_policy(self):
        Settings.config.set("policy", "learning", 'False')
        Settings.config.set("policy", "startwithhello", "True")
        Settings.config.set("summaryacts", "has_control", "False")
        if self.policy_type == "gp":
            Settings.config.set("gpsarsa", "scale", "1")

        simulator = SimulationSystem()
        simulator.run_dialogs(self.eval_batch_size)

    def train_batch(self):
        Settings.config.set("policy", "learning", 'True')
        Settings.config.set("policy", "startwithhello", "True")
        Settings.config.set("summaryacts", "has_control", "True")
        if self.policy_type == "gp":
            Settings.config.set("gpsarsa", "scale", "3")

        simulator = SimulationSystem()
        simulator.run_dialogs(self.train_batch_size)


if __name__ == '__main__':

    args = DRLPOptParser()
    config = args.config

    model = DRLP(config)
    if args.mode == 'train':
        model.train_policy()
