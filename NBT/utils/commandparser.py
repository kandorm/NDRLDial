import argparse


def NBTOptParser():

    parser = argparse.ArgumentParser(
        description='Default NBT opt parser.')

    parser.add_argument('-mode', help='modes: train|test|track')
    parser.add_argument('-config', help='config file to set.')

    return parser.parse_args()
