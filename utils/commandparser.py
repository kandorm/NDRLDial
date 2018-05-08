import argparse


def NDRLDialOptParser():

    parser = argparse.ArgumentParser(
        description='Default NDRLDial opt parser.')

    parser.add_argument('-mode', help='modes: test|track')
    parser.add_argument('-config', help='config file to set.')

    return parser.parse_args()
