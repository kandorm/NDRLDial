import re
import string


class DiaActItem(object):
    """
    Dialogue act specification
    :param slot: slot name
    :type slot: str
    :param op: comparative operation, e.g. '=' or '!='
    :type op: str or None
    :param val: value name
    :type val: str or None
    """
    def __init__(self, slot, op, val):
        self.slot = slot
        self.op = op
        self.val = val
        if val is not None and type(val) in [str, unicode] and len(val) > 0 and val[0] not in string.punctuation:
            self.val = val.lower()

    def match(self, other):
        """
        Commutative operation for comparing two items.
        Note that "self" is the goal constraint, and "other" is from the system action.
        The item in "other" must be more specific. For example, the system action confirm(food=dontcare) doesn't match
        the goal with food=chinese, but confirm(food=chinese) matches the goal food=dontcare.

        If slots are different, return True.

        If slots are the same, (possible values are x, y, dontcare, !x, !y, !dontcare)s
            x, x = True
            x, y = False
            dontcare, x = True
            x, dontcare = False
            dontcare, dontcare = True

            x, !x = False
            x, !y = True
            x, !dontcare = True
            dontcare, !x = False
            dontcare, !dontcare = False

            !x, !x = True
            !x, !y = True
            !x, !dontcare = True
            !dontcare, !dontcare = True

        :param other:
        :return:
        """
        if self.slot != other.slot:
            return True

        if self.val is None or other.val is None:
            print 'None value is given in comparison between two DiaActItem'

        if self.op == '=' and other.op == '=':
            if self.val == other.val:
                return True
            if self.val == 'dontcare':
                return True
            return False

        elif self.op == '=' and other.op == '!=':
            if self.val == 'dontcare':
                return False
            elif self.val == other.val:
                return False
            else:
                return True

        elif self.op == '!=' and other.op == '=':
            if other.val == 'dontcare':
                return False
            elif self.val == other.val:
                return False
            else:
                return True

        else:    # self.op == !=' and other.op == '!=':
            return True

    def __eq__(self, other):
        if self.slot == other.slot and self.op == other.op and self.val == other.val:
            return True
        return False

    def __hash__(self):
        return hash(repr((self.slot, self.op, self.val)))

    def __str__(self):
        return repr((self.slot, self.op, self.val))

    def __repr__(self):
        return repr((self.slot, self.op, self.val))


class DiaAct(object):
    """
    Dialogue act class.

    DiaAct = ``{'act': acttype,'items': [(slot1, op1, value1), ..])}``

    :param act_str: dialogue act in string
    """
    def __init__(self, act_str):
        # parsed is a dict of lists. 'slots' list contains dact items
        parsed = self._parse_act(act_str)
        self.act = parsed['act']
        self.items = parsed['slots']

    def append(self, slot, value, negate=False):
        """
        Add item to this act avoiding duplication
        :param slot: None
        :type slot: str or None
        :param value: None
        :type value: str or None
        :param negate: semantic operation is negation or not
        :type negate: bool [Default=False]
        :return:
        """
        op = '='
        if negate:
            op = '!='
        self.items.append(DiaActItem(slot, op, value))

    def contains_slot(self, slot):
        """
        :param slot: slot name
        :type slot: str
        :returns: (bool) answering whether self.items mentions slot
        """
        for item in self.items:
            if slot == str(item.slot):
                return True
        return False

    def contains(self, slot, value, negate=False):
        """
        :param slot: None
        :type slot: str
        :param value: None
        :type value: str
        :param negate: None
        :type negate: bool - default False
        :returns: (bool) is full semantic act in self.items?
        """
        op = '='
        if negate:
            op = '!='
        item = DiaActItem(slot, op, value)
        return item in self.items

    ########################################
    # Used for evaluation
    ########################################
    def get_value(self, slot, negate=False):
        """
        :param slot: slot name
        :type slot: str
        :param negate: relating to semantic operation, i.e slot = or slot !=.
        :type negate: bool - default False
        :returns: (str) value
        """
        value = None
        for item in self.items:
            if slot == item.slot and negate == (item.op == '!='):
                if value is not None:
                    print 'DiaAct contains multiple values for one slot: ' + str(self)
                else:
                    value = item.val
        return value

    def get_values(self, slot, negate=False):
        """
        :param slot: slot name
        :type slot: str
        :param negate: - semantic operation
        :type negate: bool - default False
        :returns: (list) values in self.items
        """
        values = []
        for item in self.items:
            if slot == item.slot and negate == (item.op == '!='):
                values.append(item.val)
        return values

    def has_conflicting_value(self, constraints):
        """
        :param constraints: as  [(slot, op, value), ...]
        :type constraints: list
        :return: (bool) True if this DiaAct has values which conflict with the given constraints. Note that consider
                 only non-name slots.
        """
        for const in constraints:
            slot = const.slot
            op = const.op
            value = const.val
            if slot == 'name' or value == 'dontcare':
                continue

            this_value = self.get_value(slot, negate=False)
            if op == '!=':
                if this_value in [value]:
                    return True
            elif op == '=':
                if this_value not in [None, value]:
                    return True
            else:
                exit('unknown constraint operator exists: ' + str(const))
        return False

    def _parse_act(self, act_str):
        result = {'act': 'null', 'slots': []}
        if act_str == "BAD ACT!!":
            return result

        m = re.search('^([^\(\)]*)\((.*)\)$', act_str.strip())
        if not m:
            return result

        result['act'] = m.group(1).strip()
        content = m.group(2)
        while len(content) > 0:
            m = re.search('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', content)
            if m:
                slot = m.group(1).strip()
                op = m.group(2).strip()
                val = m.group(3).strip("' ")
                item = DiaActItem(slot, op, val)
                content = re.sub('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', '', content)
                result['slots'].append(item)
                continue

            m = re.search('^([^,=]*)(!?=)\s*([^,]*)\s*,?', content)
            if m:
                slot = m.group(1).strip()
                op = m.group(2).strip()
                val = m.group(3).strip("' ")
                item = DiaActItem(slot, op, val)
                content = re.sub('^([^,=]*)(!?=)\s*([^,]*)\s*,?', '', content)
                result['slots'].append(item)
                continue

            m = re.search('^([^,]*),?', content)
            if m:
                slot = m.group(1).strip()
                op = None
                val = None
                item = DiaActItem(slot, op, val)
                content = re.sub('^([^,]*),?', '', content)
                result['slots'].append(item)
                continue
            raise RuntimeError('Cant parse content fragment: %s' % content)

        return result

    def to_string(self):
        """
        :returns: (str) semantic act
        """
        s = ''
        s += self.act + '('
        for i, item in enumerate(self.items):
            if i != 0:
                s += ','
            if item.slot is not None:
                s += item.slot
            if item.val is not None:
                s = s+item.op+'"'+str(item.val)+'"'
        s += ')'
        return s

    def __eq__(self, other):
        """
        :param other:
        :return: True if this DiaAct is equivalent to other. Items can be reordered.
        """
        if self.act != other.act:
            return False
        if len(self.items) != len(other.items):
            return False
        for i in self.items:
            if i not in other.items:
                return False
        for i in other.items:
            if i not in self.items:
                return False
        return True

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()


def __parse_act(act_str):
    result = {'act': 'null', 'slots': []}

    if act_str == "BAD ACT!!":
        return result

    m = re.search('^([^\(\)]*)\((.*)\)$', act_str.strip())
    if not m:
        return result

    result['act'] = m.group(1)
    content = m.group(2)
    while len(content) > 0:
        m = re.search('^([^,=]*)=\s*\"([^\"]*)\"\s*,?', content)
        if m:
            slot = m.group(1).strip()
            val = m.group(2).strip("' ")
            content = re.sub('^([^,=]*)=\s*\"([^\"]*)\"\s*,?', '', content)
            result['slots'].append([slot, val])
            continue
        m = re.search('^([^,=]*)=\s*([^,]*)\s*,?', content)
        if m:
            slot = m.group(1).strip()
            val = m.group(2).strip("' ")
            content = re.sub('^([^,=]*)=\s*([^,]*)\s*,?', '', content)
            result['slots'].append([slot, val])
            continue
        m = re.search('^([^,]*),?', content)
        if m:
            slot = m.group(1).strip()
            val = None
            content = re.sub('^([^,]*),?', '', content)
            result['slots'].append([slot, val])
            continue
        raise RuntimeError('Cant parse content fragment: %s' % content)

    for slot_pair in result['slots']:
        if slot_pair[1] is None:
            continue
        slot_pair[1] = slot_pair[1].lower()
        if slot_pair[0] == "count":
            try:
                int_value = int(slot_pair[1])
                slot_pair[1] = int_value
            except ValueError:
                pass
    return result


def _parse_act(raw_act_text, user=True):
    raw_act = __parse_act(raw_act_text)
    final_dialog_act = []

    if raw_act['act'] == "select" and user:
        raw_act['act'] = "inform"

    main_act_type = raw_act['act']

    if raw_act['act'] == "request" or raw_act['act'] == "confreq":
        for requested_slot in [slot for slot, value in raw_act['slots'] if value is None]:
            final_dialog_act.append({
                'act': 'request',
                'slots': [['slot', requested_slot]],
            })

        if raw_act['act'] == "confreq":
            main_act_type = "impl-conf"
        else:
            main_act_type = "inform"
    elif (raw_act['act'] in ['negate', 'repeat', 'affirm', 'bye', 'restart', 'reqalts', 'hello', 'silence', 'thankyou',
                             'ack', 'help', 'canthear', 'reqmore']):
        if raw_act['act'] == "hello" and not user:
            raw_act['act'] = "welcomemsg"
        final_dialog_act.append({
            'act': raw_act['act'],
            'slots': [],
        })
        main_act_type = 'inform'
    elif raw_act['act'] not in ['inform', 'deny', 'confirm', 'select', 'null', 'badact']:
        print raw_act_text
        print raw_act
        raise RuntimeError('Dont know how to convert raw act type %s' % (raw_act['act']))

    if raw_act['act'] == "confirm" and not user:
        main_act_type = "expl-conf"

    if raw_act['act'] == "select" and not user and "other" in [v for s, v in raw_act['slots']]:
        main_act_type = "expl-conf"

    if raw_act['act'] == "deny" and len(raw_act["slots"]) == 0:
        final_dialog_act.append({
            'act': "negate",
            'slots': [],
        })
    # Remove task=none
    # canthelps:
    if ["name", "none"] in raw_act["slots"] and not user:
        other_slots = []
        only_names = []  # collect the names that are the only such venues
        for slot, value in raw_act["slots"]:
            if value == "none" or slot == "other":
                continue
            if slot == "name!":
                only_names.append(value)
                continue
            other_slots.append([slot, value])

        return [{
            'act': "canthelp",
            'slots': other_slots,
        }] + [{'act': "canthelp.exception", "slots": [["name", name]]} for name in only_names]

    elif not user and "none" in [v for _, v in raw_act["slots"]]:
        if raw_act["act"] != "inform":
            return [{"act": "repeat", "slots": []}]
        none_slots = [s for s, v in raw_act["slots"] if v == "none"]
        name_value, = [v for s, v in raw_act["slots"] if s == "name"]
        other_slots = [[slot, value] for slot, value in raw_act["slots"] if value != "none"]
        final_dialog_act.append({
            'act': 'canthelp.missing_slot_value',
            'slots': [['slot', none_slot] for none_slot in none_slots] + [['name', name_value]]
        })
        if other_slots:
            raw_act = ({'act': 'inform', 'slots': other_slots})
        else:
            raw_act = {"slots": [], "act": "inform"}

    # offers
    if "name" in [slot for slot, value in raw_act["slots"]] and not user:
        name_value = [value for slot, value in raw_act["slots"] if slot == "name"]
        other_slots = [[slot, value] for slot, value in raw_act["slots"] if slot != "name"]

        final_dialog_act.append({
            'act': "offer",
            'slots': [["name", name_value]]
        })
        raw_act['slots'] = other_slots

    # group slot values by type
    # try to group date and time into inform acts
    # put location fields in their own inform acts
    main_act_slots_dict = {}
    for (raw_slot_name, raw_slot_val) in raw_act['slots']:
        slot_name = raw_slot_name
        slot_val = raw_slot_val
        slot_group = slot_name
        if slot_group not in main_act_slots_dict:
            main_act_slots_dict[slot_group] = {}
        if slot_name not in main_act_slots_dict[slot_group]:
            main_act_slots_dict[slot_group][slot_name] = []
        if slot_val not in main_act_slots_dict[slot_group][slot_name]:
            main_act_slots_dict[slot_group][slot_name].append(slot_val)

    for slot_group_name, slot_group_items in main_act_slots_dict.items():
        for slot, vals in slot_group_items.items():
            # if slot in ["task", "type"] :
            #     continue
            # we shouldn't skip this
            if slot == "":
                slot = "this"
            if main_act_type == "deny" and len(vals) == 2 and "dontcare" not in vals:
                # deal with deny(a=x, a=y)
                false_value = vals[0]
                true_value = vals[1]
                final_dialog_act.append({
                    'act': "deny",
                    'slots': [[slot, false_value]],
                })
                final_dialog_act.append({
                    'act': "inform",
                    'slots': [[slot, true_value]],
                })
            else:
                for val in vals:

                    if val is None or val == "other":
                        continue

                    if len(slot) > 0 and slot[-1] == "!":
                        slot = slot[:-1]
                        slots = [[slot, val]]
                        final_dialog_act.append({
                            'act': "deny",
                            'slots': slots,
                        })
                    else:
                        slots = [[slot, val]]
                        if ((slot, val) == ("this", "dontcare")) and (main_act_type != "inform"):
                            continue

                        final_dialog_act.append({
                            'act': ("inform" if slot == "count" else main_act_type),
                            'slots': slots,
                        })

    if not user and len(final_dialog_act) == 0:
        final_dialog_act.append({"act": "repeat", "slots": []})
    return final_dialog_act


def parse_act(raw_act_text, user=True):
    final = []
    for act_text in raw_act_text.split("|"):
        try:
            final += _parse_act(act_text, user=user)
        except RuntimeError:
            pass  # add nothing to final if junk act recieved
    return final


def infer_slots_for_act(uacts, ontology=None):
    """
    Works out the slot from the ontology and value

    :param uacts:
    :param ontology:
    :return: user's dialogue acts
    """
    for uact in uacts:
        for index in range(len(uact["slots"])):
            (slot, value) = uact["slots"][index]
            if slot == "this" and value != "dontcare":
                skip_this = False
                if ontology:
                    for s, vals in ontology["informable"].iteritems():
                        if value in vals:
                            if slot != "this":  # Already changed!
                                print "Warning: " + value + " could be for " + slot + " or " + s
                                skip_this = True
                            slot = s
                else:
                    slot = "type"  # default
                if not skip_this:
                    if slot == "this":
                        print "Warning: unable to find slot for value " + value
                        uact["slots"][index] = ("", "")
                    else:
                        uact["slots"][index] = (slot, value)
            uact["slots"][:] = list((s, v) for (s, v) in uact["slots"] if s != "" or v != "")
    return uacts
