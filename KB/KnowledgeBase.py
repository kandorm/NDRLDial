import KBManager

global_kb = None


def init_global_kb():
    global global_kb
    global_kb = KBManager.KBManager()
