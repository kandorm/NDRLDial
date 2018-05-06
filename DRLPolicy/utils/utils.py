import os


# Saving policy:
def check_dir_exists_and_make(full_path):
    """
    Used when saving a policy -- if dir doesn't exisit --> is created
    """
    if '/' in full_path:
        path = full_path.split('/')
        path = '/'.join(path[:-1])
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
