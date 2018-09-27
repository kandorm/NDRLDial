'''
NBT.py - Main script to train and test Neural Belief Tracker.
===============================================================

**Basic Execution**
    >>> python main.py -config CONFIG -mode [train|test|track]

    import :mod:'semanticbelieftracking.NBT.utils.commandparser' |.|
************************
'''

import sys
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

sys.path.insert(0, '.')
from NBT.utils.commandparser import NBTOptParser
from NBT.tracker.net import Tracker

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


if __name__ == '__main__':

    args = NBTOptParser()
    config = args.config

    model = Tracker(config)
    if args.mode == 'train':
        model.train_net()
    elif args.mode == 'test':
        model.test_net()
    else:
        previous_belief_state = None

        while True:
            utterance = raw_input("Enter utterance for prediction:")
            prediction_dict, previous_belief_state, _ = \
                model.track_utterance([(utterance, 1.0)], [""], [""], [""], previous_belief_state)
            print json.dumps(prediction_dict, indent=4)


