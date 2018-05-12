import argparse


def DRLPOptParser():

    parser = argparse.ArgumentParser(
        description='Default DRLP opt parser.')

    parser.add_argument('-mode', help='modes: train')
    parser.add_argument('-config', help='config file to set.')

    return parser.parse_args()
