import numpy.random as nprandom
import ConfigParser

config = None
random = None


def init(config_file, seed=None):
    '''
    Called by dialog programs (simulate, texthub, dialogueserver) to init Settings globals
    '''
    # Config:
    # -----------------------------------------
    load_config(config_file)

    # Seed:
    # -----------------------------------------
    if seed is None:
        if config.has_option("general", 'seed'):
            seed = config.getint("general", "seed")
    seed = set_seed(seed)
    config.set('general', 'seed', str(seed))
    return seed


def load_config(config_file):
    global config
    config = None
    if config_file is not None:
        try:
            config = ConfigParser.ConfigParser()
            config.read(config_file)
        except Exception as inst:
            print "Failed to parse file", inst
    else:
        # load empty config
        config = ConfigParser.ConfigParser()


def set_seed(seed):
    global random
    if seed is None:
        random1 = nprandom.RandomState(None)
        seed = random1.randint(1000000000)
    random = nprandom.RandomState(seed)

    return seed
