import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from NDRLDial import NDRLDial
from utils.commandparser import NDRLDialOptParser


if __name__ == '__main__':

    args = NDRLDialOptParser()
    config = args.config

    dial = NDRLDial(config)
    if args.mode == 'test':
        dial.test()
    elif args.mode == 'track':
        belief_state = {}
        last_sys_act = ''
        generated = ''
        while True:
            utterance = raw_input(">> User Input: ")
            response = dial.reply(utterance, last_sys_act, belief_state)
            belief_state = response['belief_state']
            last_sys_act = response['last_sys_act']
            print '>> Sys reply:', response['generated']
