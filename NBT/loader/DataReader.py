# -*- coding: utf-8 -*-
from copy import deepcopy
import json
import random
import numpy
import string

from NBT.utils.wvecUtil import *


class DataReader(object):

    def __init__(self, wvecfile, ontologyfile, language,
                 trainfile, testfile, validfile, percentage):

        self.language = language
        self.percentage = percentage
        self.data = {'train': (), 'valid': (), 'test': ()}

        # loading files
        self.dialogue_ontology = json.load(codecs.open(ontologyfile, "r", "utf-8"))["informable"]
        self._setup_data(trainfile, testfile, validfile)

        # prepare word vectors
        self.word_vectors = self._load_word_vectors(wvecfile)
        self.word_vector_size = random.choice(self.word_vectors.values()).shape[0]
        self._prepare_slot_value_vectors()

    def _load_word_vectors(self, wvecfile):
        if not os.path.isfile(wvecfile):
            download_word_vectors(wvecfile)
        word_vectors = load_word_vectors(wvecfile, primary_language=self.language)
        word_vectors[u"tag-slot"] = xavier_vector(u"tag-slot")
        word_vectors[u"tag-value"] = xavier_vector(u"tag-value")

        # a bit of hard-coding to make our lives easier.
        if u"price" in word_vectors and u"range" in word_vectors:
            word_vectors[u"pricerange"] = word_vectors[u"price"] + word_vectors[u"range"]
            word_vectors[u"price range"] = word_vectors[u"price"] + word_vectors[u"range"]
        if u"post" in word_vectors and u"code" in word_vectors:
            word_vectors[u"postcode"] = word_vectors[u"post"] + word_vectors[u"code"]
        if u"dont" in word_vectors and u"care" in word_vectors:
            word_vectors[u"dontcare"] = word_vectors[u"dont"] + word_vectors[u"care"]
        if u"addresses" in word_vectors:
            word_vectors[u"addressess"] = word_vectors[u"addresses"]
        if u"dont" in word_vectors:
            word_vectors[u"don't"] = word_vectors[u"dont"]

        if self.language == "italian":
            word_vectors[u"dontcare"] = word_vectors[u"non"] + word_vectors[u"importa"]
            word_vectors[u"non importa"] = word_vectors[u"non"] + word_vectors[u"importa"]

        if self.language == "german":
            word_vectors[u"dontcare"] = word_vectors[u"es"] + word_vectors[u"ist"] + word_vectors[u"egal"]
            word_vectors[u"es ist egal"] = word_vectors[u"es"] + word_vectors[u"ist"] + word_vectors[u"egal"]

        return word_vectors

    def _prepare_slot_value_vectors(self):

        # add dontcare value into informable slot
        # add word vector of value (not in the pre-trained word vectors)
        dontcare_value = "dontcare"
        if self.language == "italian":
            dontcare_value = "non importa"
        if self.language == "german":
            dontcare_value = "es ist egal"

        slots = self.dialogue_ontology.keys()
        for slot in slots:
            if dontcare_value not in self.dialogue_ontology[slot] and slot != "request":
                self.dialogue_ontology[slot].append(dontcare_value)
            for value in self.dialogue_ontology[slot]:
                value = unicode(value)
                if u" " not in value and value not in self.word_vectors:
                    self.word_vectors[value] = xavier_vector(value)
                    print "  ---Generating word vector for:", \
                        value.encode("utf-8"), ":::", numpy.sum(self.word_vectors[value])

        # add up multi-word word values to get their representation:
        for slot in slots:
            if " " in slot:
                slot = unicode(slot)
                self.word_vectors[slot] = numpy.zeros((self.word_vector_size,), dtype="float32")
                constituent_words = slot.split()
                for word in constituent_words:
                    word = unicode(word)
                    if word in self.word_vectors:
                        self.word_vectors[slot] += self.word_vectors[word]

            for value in self.dialogue_ontology[slot]:
                if " " in value:
                    value = unicode(value)
                    self.word_vectors[value] = numpy.zeros((self.word_vector_size,), dtype="float32")
                    constituent_words = value.split()
                    for word in constituent_words:
                        word = unicode(word)
                        if word in self.word_vectors:
                            self.word_vectors[value] += self.word_vectors[word]

    def _normalize_dial(self, dial):
        """
        Returns a list of (tuple, belief_state) for each turn in the dialogue.
        i.e. list of ( (transcription, asr_obs),
                        cur_sysreq,         # current system request
                        cur_sysconf_slot,   # current system confirm slot
                        cur_sysconf_value,  # current system confirm value
                        cur_bs,             # current belief state
                        prev_bs,            # previous belief state
                        turn_label)         # turn label
        """
        dialogue_representation = []

        # initial belief state
        # belief state to be given at each turn
        null_bs = {}
        informable_slots = []
        pure_requestables = []
        if self.language == "english":
            null_bs["food"] = "none"
            null_bs["pricerange"] = "none"
            null_bs["area"] = "none"
            null_bs["request"] = []
            informable_slots = ["food", "pricerange", "area"]
            pure_requestables = ["address", "phone", "postcode"]
        elif self.language == "italian":
            null_bs["area"] = "none"
            null_bs["cibo"] = "none"
            null_bs["prezzo"] = "none"
            null_bs["request"] = []
            informable_slots = ["cibo", "prezzo", "area"]
            pure_requestables = ["codice postale", "telefono", "indirizzo"]
        elif self.language == "german":
            null_bs["gegend"] = "none"
            null_bs["essen"] = "none"
            null_bs["preisklasse"] = "none"
            null_bs["request"] = []
            informable_slots = ["essen", "preisklasse", "gegend"]
            pure_requestables = ["postleitzahl", "telefon", "adresse"]

        prev_belief_state = deepcopy(null_bs)

        for idx, turn in enumerate(dial):
            cur_DA = turn["system_acts"]
            cur_sysreq = []
            cur_sysconf_slot = []
            cur_sysconf_value = []

            for each_da in cur_DA:
                if each_da in informable_slots:
                    cur_sysreq.append(each_da)
                elif each_da in pure_requestables:
                    cur_sysconf_slot.append("request")
                    cur_sysconf_value.append(each_da)
                else:
                    if type(each_da) is list:
                        cur_sysconf_slot.append(each_da[0])
                        cur_sysconf_value.append(each_da[1])
            if not cur_sysreq:
                cur_sysreq = [""]
            if not cur_sysconf_slot:
                cur_sysconf_slot = [""]
                cur_sysconf_value = [""]

            cur_transcription = normalize_transcription(turn["transcript"], self.language)
            cur_asr = [(normalize_transcription(hyp, self.language), score) for (hyp, score) in turn["asr"]]
            cur_labels = turn["turn_label"]
            cur_bs = deepcopy(prev_belief_state)
            turn_bs = deepcopy(null_bs)

            # reset requestables at each turn
            if "request" in prev_belief_state:
                del prev_belief_state["request"]
            cur_bs["request"] = []

            for label in cur_labels:
                c_slot, c_value = label
                if c_slot in informable_slots:
                    cur_bs[c_slot] = c_value
                    turn_bs[c_slot] = c_value
                elif c_slot == "request":
                    cur_bs["request"].append(c_value)
                    turn_bs["request"].append(c_value)

            dialogue_representation.append((
                (cur_transcription, cur_asr),
                cur_sysreq,
                cur_sysconf_slot,
                cur_sysconf_value,
                deepcopy(cur_bs),
                deepcopy(prev_belief_state),
                deepcopy(turn_bs)
            ))

            prev_belief_state = deepcopy(cur_bs)

        return dialogue_representation

    def _load_data(self, filepath):
        """
        This method loads dataset as a collection of utterances.
        """
        dialogues = []
        training_turns = []

        data = json.load(codecs.open(filepath, "r", "utf-8"))
        count = int(float(len(data)) * float(self.percentage))
        print "Loading form", filepath, "percentage is:", self.percentage, "so loading:", count
        for idx in range(count):
            cur_dial = self._normalize_dial(data[idx]["dialogue"])
            dialogues.append(cur_dial)

            # Informable slot set 'none' and request slot set []
            for turn_idx, turn in enumerate(cur_dial):
                cur_label = [("request", req_slot) for req_slot in turn[4]["request"]]
                for inf_slot in turn[4]:
                    if inf_slot != "request":
                        cur_label.append((inf_slot, turn[4][inf_slot]))

                turn_label = [("request", req_slot) for req_slot in turn[6]["request"]]
                for inf_slot in turn[6]:
                    if inf_slot != "request":
                        turn_label.append((inf_slot, turn[6][inf_slot]))

                transcription_and_asr = turn[0]
                cur_utterance = (transcription_and_asr, turn[1], turn[2], turn[3], cur_label, turn[5], turn_label) # turn[5] is the past belief state

                training_turns.append(cur_utterance)

        return dialogues, training_turns

    def _setup_data(self, trainfile, testfile, validfile):
        # load data from file
        self.data['train'] = self._load_data(trainfile)
        self.data['test'] = self._load_data(testfile)
        self.data['valid'] = self._load_data(validfile)


def normalize_transcription(transcription, language):
    """
    Returns the clean (i.e. handling interpunction signs) string for the given language.
    """
    exclude = set(string.punctuation)
    exclude.remove("'")

    transcription = ''.join(ch for ch in transcription if ch not in exclude)

    transcription = transcription.lower()
    transcription = transcription.replace(u"’", "'")
    transcription = transcription.replace(u"‘", "'")
    transcription = transcription.replace("don't", "dont")
    if language == "italian":
        transcription = transcription.replace("'", " ")
    if language == "english":
        transcription = transcription.replace("'", "")

    return transcription
