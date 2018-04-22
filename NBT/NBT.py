'''
NBT.py - Main script to train and test Neural Belief Tracker.
===============================================================

**Basic Execution**
    >>> python NBT -config CONFIG -mode [train|test|track]

    import :mod:'semanticbelieftracking.NBT.utils.commandparser' |.|
************************
'''

import sys
import json

sys.path.insert(0, '.')
from NBT.utils.commandparser import NBTOptParser
from NBT.tracker.net import Model

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


if __name__ == '__main__':

    args = NBTOptParser()
    config = args.config

    model = Model(config)
    if args.mode == 'train':
        model.train_net()
    elif args.mode == 'test':
        model.test_net()
    else:
        previous_belief_state = None

        while True:
            utterance = raw_input("Enter utterance for prediction:")
            predictions = model.track_utterance([(utterance, 1.0)], [""], [""], [""], previous_belief_state)
            print json.dumps(predictions, indent=4)


