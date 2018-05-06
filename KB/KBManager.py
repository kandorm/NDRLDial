import DataBaseSQLite


class KBManager(object):

    def __init__(self, db_file=None):
        if db_file is None:
            db_file = 'KB/db/CamRestaurants-dbase.db'
        try:
            self.db = DataBaseSQLite.DataBaseSQLite(db_file)
        except Exception as e:
            print e

    def entity_by_features(self, constraints):
        if self.db is not None:
            return self.db.entity_by_features(constraints=constraints)
        return {}

    def get_length_entity_by_features(self, constraints):
        if self.db is not None:
            return self.db.get_length_entity_by_features(constraints=constraints)
        return 0
