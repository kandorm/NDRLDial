import sqlite3
import numpy as np


class DataBaseSQLite(object):

    def __init__(self, db_file):
        self.random_ten_entities_query = 'select * from CamRestaurants ORDER BY RANDOM() limit 10'

        self.cursor = self._load_db(db_file)

    def _load_db(self, db_file):
        """
        Connect database.

        :param db_file: database file path
        :return: connection of data base
        """
        cursor = None
        try:
            db_connection = sqlite3.connect(db_file)
            db_connection.row_factory = self._dict_factory
            cursor = db_connection.cursor()
        except Exception as e:
            print e
        return cursor

    def _dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def entity_by_features(self, constraints):
        """
        Retrieves from database all entities matching the given constraints.

        :param constraints: features. Dict {slot:value, ...} or List [(slot, op, val), ...] \
        (NB. the tuples in the list are actually a :class:`dact` instances)
        :returns: (list) all entities (each a dict)  matching the given features.
        """
        # 1. Format constraints into sql_query
        # NO safety checking - constraints should be a list or a dict
        # Also no checking of values regarding none:   if const.val == [None, '**NONE**']: --> ERROR
        do_rand = False

        if len(constraints):
            bits = []
            values = []
            if isinstance(constraints, list):
                for cond in constraints:
                    if cond.op == '=' and cond.val == 'dontcare':
                        continue    # NB assume no != 'dontcare' case occurs - so not handling
                    if cond.val in ['none', None]:
                        continue
                    if cond.op == '!=' and cond.val != 'dontcare':
                        bits.append(cond.slot + '!= ?')
                    else:
                        bits.append(cond.slot + '= ?  COLLATE NOCASE')
                    values.append(cond.val)
            elif isinstance(constraints, dict):
                for slot, value in constraints.iteritems():
                    if value and value not in ['dontcare', 'none', None]:
                        bits.append(slot + '= ?  COLLATE NOCASE')
                        values.append(value)

            # 2. Finalise and Execute sql_query
            try:
                if len(bits):
                    sql_query = 'select * from CamRestaurants where '
                    sql_query += ' and '.join(bits)
                    self.cursor.execute(sql_query, tuple(values))
                else:
                    sql_query = self.random_ten_entities_query
                    self.cursor.execute(sql_query)
                    do_rand = True
            except Exception as e:
                print e
        else:
            # NO CONSTRAINTS --> get all entities in database?
            # TODO:: check when this occurs ... is it better to return a single, random entity? --> returning random 10

            # 2. Finalise and Execute sql_query
            sql_query = self.random_ten_entities_query
            self.cursor.execute(sql_query)
            do_rand = True

        results = self.cursor.fetchall()    # can return directly

        if do_rand:
            np.random.shuffle(results)
        return results

    def get_length_entity_by_features(self, constraints):
        return len(self.entity_by_features(constraints))
