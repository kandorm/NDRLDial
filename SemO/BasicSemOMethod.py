import os
import re
import tokenize
import copy
from random import shuffle
from collections import defaultdict
from SemO.utils import Scanner
from SemO.utils.dact import DiaActItem, DiaAct


def parse_output(input_string):
    """
    Utility function used within this file's classes.
    :param input_string: None
    :type input_string: str
    """
    output_scanner = Scanner.Scanner(input_string)
    output_scanner.next()
    words = []
    prev_was_variable = False
    while output_scanner.cur[0] != tokenize.ENDMARKER:
        if output_scanner.cur[0] == tokenize.NAME:
            words.append(output_scanner.cur[1])
            output_scanner.next()
            prev_was_variable = False
        elif output_scanner.cur[1] == '$':
            variable = '$'
            output_scanner.next()
            variable += output_scanner.cur[1]
            words.append(variable)
            output_scanner.next()
            prev_was_variable = True
        elif output_scanner.cur[1] == '%':
            func = '%'
            output_scanner.next()
            while output_scanner.cur[1] != ')':
                func += output_scanner.cur[1]
                output_scanner.next()
            func += output_scanner.cur[1]
            words.append(func)
            output_scanner.next()
            prev_was_variable = True
        else:
            if prev_was_variable:
                words.append(output_scanner.cur[1])
                output_scanner.next()
            else:
                words[-1] += output_scanner.cur[1]
                output_scanner.next()
            prev_was_variable = False
    return words


class SemO(object):
    """
    Interface class for a language generator.
    Responsible for generating a natural language sentence from a dialogue act representation.
    To create your own SemO methods, derive from this class.
    """
    def generate(self, act):
        """
        Main generation method: mapping from system act to natural language
        :param act: the system act to generate
        :type act: str
        :returns: the natural language realisation of the given system act
        """
        pass


class BasicSemO(SemO):
    """
    Template-based output generator.  Note that the class inheriting from object is important - without this the super
    method can not be called -- This relates to 'old-style' and 'new-style' classes in python if interested ...
    """
    def __init__(self):
        template_path = 'SemO/templates/CamRestaurantsMessages.txt'
        self.generator = BasicTemplateGenerator(template_path)

    def generate(self, act):
        return self.generator.transform(act)


class BasicTemplateGenerator(object):
    """
    The basic template generator loads a list of template-based rules from a string.
    These are then applied on any input dialogue act and used to generate an output string.

    :param filename: the template rules file
    :type filename: str
    """
    def __init__(self, filename):
        if os.path.exists(filename):
            f = open(filename)
            string = f.read()
            string.replace('\t', ' ')
            string_without_comment = Scanner.remove_comments(string)
            scanner = Scanner.Scanner(string_without_comment)
            scanner.next()
            self.rules = []
            self.functions = []
            self.function_map = {}
            self._parse_rules(scanner)
            f.close()
        else:
            exit("Cannot locate template file {}".format(filename))

    def transform(self, sys_act):
        """
        Transforms the sysAct from a semantic utterance form to a text form using the rules in the generator.
        This function will run the sysAct through all variable rules and will choose the best one according to the
        number of matched act types, matched items and missing items.

        :param sys_act: input system action (semantic form).
        :type sys_act: str
        :returns: (str) natural language
        """
        input_utt = DiaAct(sys_act)

        # transform system acts with slot op "!=" to "="
        # and add slot-value pair other=true which is needed by NLG rule base
        # assumption: "!=" only appears if there are no further alternatives, ie, inform(name=none, name!=place!, ...)
        neg_found = False
        for item in input_utt.items:
            if item.op == "!=":
                item.op = u"="
                neg_found = True
        if neg_found:
            other_true = DiaActItem(u'other', u'=', u'true')
            input_utt.items.append(other_true)

        # Iterate over BasicTemplateRule rules.
        best = None
        best_matches = 0
        best_type_match = 0
        best_missing = 1000
        best_non_term_map = None
        for rule in self.rules:
            out, matches, missing, type_match, non_term_map = rule.generate(input_utt)

            # Pick up the best rule.
            choose_this = False
            if type_match > 0:
                if missing < best_missing:
                    choose_this = True
                elif missing == best_missing:
                    if type_match > best_type_match:
                        choose_this = True
                    elif type_match == best_type_match and matches > best_matches:
                        choose_this = True

            if choose_this:
                best = out
                best_missing = missing
                best_type_match = type_match
                best_matches = matches
                best_non_term_map = non_term_map

                if best_type_match == 1 and best_missing == 0 and best_matches == len(input_utt.items):
                    break

        best = self._compute_ftn(best, best_non_term_map)
        return ' '.join(best)

    def _parse_rules(self, scanner):
        """
        Check the given rules
        :param scanner: of :class:`~SemO.utils.Scanner.Scanner`
        :type scanner: instance
        """
        try:
            while scanner.cur[0] not in [tokenize.ENDMARKER]:
                if scanner.cur[0] == tokenize.NAME:
                    self.rules.append(BasicTemplateRule(scanner))
                elif scanner.cur[1] == '%':
                    ftn = BasicTemplateFunction(scanner)
                    self.functions.append(ftn)
                    self.function_map[ftn.function_name] = ftn
                else:
                    raise SyntaxError('Expected a string or function map but got ' +
                                      scanner.cur[1] + ' at this position while parsing generation rules.')

        except SyntaxError as inst:
            print inst

    def _compute_ftn(self, input_words, non_term_map):
        """
        Applies this function to convert a function into a string.

        :param input_words: of generated words. Some words might contain function. e.g. %count_rest($X) or %$Y_str($P)
        :type input_words: list
        :param non_term_map:
        :returns: (list) modified input_words
        """
        for i, word in enumerate(input_words):
            if '%' not in word:
                continue
            m = re.search('^([^\(\)]*)\((.*)\)(.*)$', word.strip())
            if m is None:
                exit('Parsing failed in %s' % word.strip())
            ftn_name = m.group(1)
            ftn_args = [x.strip() for x in m.group(2).split(',')]
            remaining = ''
            if len(m.groups()) > 2:
                remaining = m.group(3)

            # Processing function name.
            if '$' in ftn_name:
                tokens = ftn_name.split('_')
                if len(tokens) > 2:
                    exit('More than one underbar _ found in function name %s' % ftn_name)
                var = tokens[0][1:]
                if var not in non_term_map:
                    exit('Unable to find nonterminal %s in non terminal map.' % var)
                ftn_name = ftn_name.replace(var, non_term_map[var])

            # Processing function args.
            for j, arg in enumerate(ftn_args):
                if arg[0] == '%':
                    exit('% in function argument {}'.format(str(word)))
                elif arg[0] == '$':
                    ftn_args[j] = non_term_map[arg]

            if ftn_name not in self.function_map:
                exit('Function name %s is not found.' % ftn_name)
            else:
                input_words[i] = self.function_map[ftn_name].transform(ftn_args) + remaining

        return input_words


class BasicTemplateRule(object):
    """
    The template rule corresponds to a single line in a template rules file.
    This consists of an act (including non-terminals) that the rule applies to with an output string to generate
    (again including non-terminals).
    Example::
        select(food=$X, food=dontcare) : "Sorry would you like $X food or you dont care";
         self.rue_items = {food: [$X, dontcare]}
    """
    def __init__(self, scanner):
        """
        Reads a template rule from the scanner. This should have the form 'act: string' with optional comments.
        """
        self.rule_act = self._read_from_stream(scanner)
        rule_act_str = str(self.rule_act)

        if '))' in rule_act_str:
            print 'Two )): ' + rule_act_str
        if self.rule_act.act == 'badact':
            exit('Generated bac act rule: ' + rule_act_str)

        scanner.check_token(':', 'Expected \':\' after ' + rule_act_str)
        scanner.next()
        if scanner.cur[0] not in [tokenize.NAME, tokenize.STRING]:
            raise SyntaxError('Expected string after colon')

        # Parse output string.
        self.output = scanner.cur[1].strip('"\'').strip()
        self.output_list = parse_output(self.output)

        scanner.next()
        scanner.check_token(';', 'Expected \';\' at the end of string')
        scanner.next()

        # rule_items = {slot: [val1, val2, ...], ...}
        self.rule_items = defaultdict(list)
        for item in self.rule_act.items:
            self.rule_items[item.slot].append(item.val)

    def generate(self, input_act):
        """
        Generates a text from using this rule on the given input act.
        Also edits the passed variables to store the number of matched items,
        number of missing items and number of matched utterance types.
        Note that the order of the act and rule acts must be exactly the same.

        :returns: output, match_count, missing_count, type_match_count
        """
        type_match_count = 0
        match_count = 0
        missing_count = 0
        non_term_map = {}
        if self.rule_act.act == input_act.act:
            type_match_count += 1
            match_count, missing_count, non_term_map = self._match_act(input_act)

        return self._generate_from_map(non_term_map), match_count, missing_count, type_match_count, non_term_map

    def _generate_from_map(self, non_term_map):
        """
        Does the generation by substituting values in non_term_map.

        :param non_term_map: {$X: food, ...}
        :return: list of generated words
        """
        num_missing = 0
        word_list = copy.deepcopy(self.output_list)

        for i, word in enumerate(word_list):
            if word[0] == '$':  # Variable $X
                if word not in non_term_map:
                    num_missing += 1
                else:
                    word_list[i] = non_term_map[word]
            # %$ function in output will be transformed later.

        return word_list

    def _match_act(self, act):
        """
        This function matches the given act against the slots in rule_map
        any slot-value pairs that are matched will be placed in the non-terminal map.

        :param act: The act to match against (i.e. the act that is being transformed, with no non-terminals)
        :returns (found_count, num_missing): found_count = # of items matched, num_missing = # of missing values.
        """
        non_term_map = {}  # Any mathced non-terminals are placed here.
        rules = {}
        dollar_rules = {}
        for slot in self.rule_items:
            if slot[0] == '$':
                # Copy over rules that have an unspecified slot.
                value_list = copy.deepcopy(self.rule_items[slot])
                if len(value_list) > 1:
                    exit('Non-terminal %s is mapped to multiple values %s' % (slot, str(value_list)))
                dollar_rules[slot] = value_list[0]
            else:
                # Copy over rules that have a specified slot.
                rules[slot] = copy.deepcopy(self.rule_items[slot])

        found_count = 0
        # For each item in the given system action.
        rnd_items = act.items
        shuffle(rnd_items)
        for item in rnd_items:
            found = False
            if item.slot in rules:
                if item.val in rules[item.slot]:
                    # Found this exact terminal in the rules. (e.g. food=none)
                    found = True
                    found_count += 1
                    rules[item.slot].remove(item.val)
                else:
                    # Found the rule containing the same slot but no terminal.
                    # Use the first rule which has a non-terminal.
                    val = None
                    for value in rules[item.slot]:
                        if value[0] == '$':
                            # Check if we've already assigned this non-terminal.
                            if value not in non_term_map:
                                found = True
                                val = value
                                break
                            elif non_term_map[value] == item.val:
                                # This is a non-terminal so we can re-write it if we've already got it.
                                # Then this value is the same so that also counts as found.
                                found = True
                                val = value
                                break

                    if found:
                        non_term_map[val] = item.val
                        rules[item.slot].remove(val)
                        found_count += 1

            if not found and len(dollar_rules) > 0:
                # The slot doesn't match. Just use the first dollar rule.
                for slot in dollar_rules:
                    if item.val == dollar_rules[slot]:  # $X=dontcare
                        found = True
                        non_term_map[slot] = item.slot
                        del dollar_rules[slot]
                        found_count += 1
                        break

                if not found:
                    for slot in dollar_rules:
                        if dollar_rules[slot] is not None and dollar_rules[slot][0] == '$':  # $X=$Y
                            found = True
                            non_term_map[slot] = item.slot
                            non_term_map[dollar_rules[slot]] = item.val
                            del dollar_rules[slot]
                            found_count += 1
                            break

        num_missing = len([val for sublist in rules.values() for val in sublist])
        return found_count, num_missing, non_term_map

    def _read_from_stream(self, scanner):
        sin = ''
        while scanner.cur[1] != ';' and scanner.cur[0] != tokenize.ENDMARKER and scanner.cur[1] != ':':
            sin += scanner.cur[1]
            scanner.next()
        return DiaAct(sin)

    def __str__(self):
        s = str(self.rule_act)
        s += ' : '
        s += self.output + ';'
        return s


class BasicTemplateFunction(object):
    """
    A function in the generation rules that converts a group of inputs into an output string.
    The use of template functions allows for simplification of the generation file as the way
    a given group of variables is generated can be extended over multiple rules.

    The format of the function is::

        %functionName($param1, $param2, ...) {
            p1, p2, ... : "Generation output";}

    :param scanner: of :class:`Scanner`
    :type scanner: instance
    """

    def __init__(self, scanner):
        scanner.check_token('%', 'Expected map variable name (with %)')
        scanner.next()
        self.function_name = '%' + scanner.cur[1]
        scanner.next()
        scanner.check_token('(', 'Expected open bracket ( after declaration of function')

        self.parameter_names = []
        while True:
            scanner.next()
            # print scanner.cur
            if scanner.cur[1] == '$':
                scanner.next()
                self.parameter_names.append(scanner.cur[1])
            elif scanner.cur[1] == ')':
                break
            elif scanner.cur[1] != ',':
                raise SyntaxError(
                    'Expected variable, comma, close bracket ) in input definition of template function rule')

        if len(self.parameter_names) == 0:
            raise SyntaxError('Must have some inputs in function definition: ' + self.function_name)

        scanner.next()
        scanner.check_token('{', 'Expected open brace after declaration of function ' + self.function_name)
        scanner.next()

        self.rules = []
        while scanner.cur[1] != '}':
            new_rule = BasicTemplateFunctionRule(scanner)
            if len(new_rule.inputs) != len(self.parameter_names):
                raise SyntaxError('Different numbers of parameters (%d) in rules and definition (%d) for function: %s' %
                                  (len(new_rule.inputs), len(self.parameter_names), self.function_name))
            self.rules.append(new_rule)
        scanner.next()

    def transform(self, inputs):
        """
        :param inputs: Array of function arguments.
        :returns: None
        """
        inputs = [w.replace('not available', 'none') for w in inputs]

        for rule in self.rules:
            if rule.is_applicable(inputs):
                return rule.transform(inputs)

        exit('In function %s: No rule to transform inputs %s' % (self.function_name, str(inputs)))


class BasicTemplateFunctionRule(object):
    """
    A single line of a basic template function. This does a conversion of a group of values into a string.
    e.g. p1, p2, ... : "Generation output"

    :param scanner: of :class:`Scanner`
    :type scanner: instance
    """
    def __init__(self, scanner):
        """
        Loads a template function rule from the stream.
        The rule should have the format:
            input1, input2 : "output string";
        """
        self.inputs = []
        self.input_map = {}
        while True:
            # print scanner.cur
            if scanner.cur[1] == '$' or scanner.cur[0] in [tokenize.NUMBER, tokenize.STRING, tokenize.NAME]:
                input = scanner.cur[1]
                if scanner.cur[1] == '$':
                    scanner.next()
                    input += scanner.cur[1]
                # Add to lookup table.
                self.input_map[input] = len(self.inputs)
                self.inputs.append(input.strip('"\''))
                scanner.next()
            elif scanner.cur[1] == ':':
                scanner.next()
                break
            elif scanner.cur[1] == ',':
                scanner.next()
            else:
                raise SyntaxError('Expected string, comma, or colon in input definition of template function rule.')

        if len(self.inputs) == 0:
            raise SyntaxError('No inputs specified for template function rule.')

        # Parse output string.
        scanner.check_token(tokenize.STRING, 'Expected string output for template function rule.')
        self.output = scanner.cur[1].strip('\"').strip()
        self.output = parse_output(self.output)

        scanner.next()
        scanner.check_token(';', 'Expected semicolon to end template function rule.')
        scanner.next()

    def __str__(self):
        return str(self.inputs) + ' : ' + str(self.output)

    def is_applicable(self, inputs):
        """
        Checks if this function rule is applicable for the given inputs.

        :param inputs: array of words
        :returns: (bool)
        """
        if len(inputs) != len(self.inputs):
            return False

        for i, word in enumerate(self.inputs):
            if word[0] != '$' and inputs[i] != word:
                return False

        return True

    def transform(self, inputs):
        """
        Transforms the given inputs into the output. All variables in the output list are looked up in the map
        and the relevant value from the inputs is chosen.

        :param inputs: array of words.
        :returns: Transformed string.
        """
        result = []
        for output_word in self.output:
            if output_word[0] == '$':
                if output_word not in self.input_map:
                    exit('Could not find variable %s' % output_word)
                result.append(inputs[self.input_map[output_word]])
            else:
                result.append(output_word)
        return ' '.join(result)
