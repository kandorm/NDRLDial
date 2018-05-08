import sys
import time
import numpy
import math
import json

from NBT.utils.wvecUtil import *
from NBT.loader.DataReader import *
from NBT.nn.NeuralBeliefTracker import *

from ConfigParser import SafeConfigParser


class Tracker(object):

    def __init__(self, config=None):
        # not enough info to execute
        if config is None:
            print "Please specify config file ..."
            return

        print '\n\ninitialize word vectors and model variables...'

        # config parser
        parser = SafeConfigParser()
        parser.read(config)

        # setting training mode
        self.debug                     = parser.getboolean('train', 'debug')
        if self.debug:
            print 'loading settings from config file ...'
        self.batches_per_epoch         = parser.getint('train', 'batches_per_epoch')
        self.max_epoch                 = parser.getint('train', 'max_epoch')
        self.batch_size                = parser.getint('train', 'batch_size')
        # setting file path
        self.modeldir                 = parser.get('file', 'model')
        if self.modeldir[-1] != '/':
            self.modeldir = self.modeldir + "/"
        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)
        self.resultfile                = parser.get('file', 'result')
        self.wvecfile                  = parser.get('file', 'wvec')
        self.ontologyfile              = parser.get('file', 'ontology')
        self.trainfile                 = parser.get('file', 'train')
        self.testfile                  = parser.get('file', 'test')
        self.validfile                 = parser.get('file', 'valid')
        # setting data paragram
        self.percentage                = float(parser.getfloat('data', 'percentage')) / 100.0
        # setting model paragram
        self.language                  = parser.get('model', 'language')
        self.longest_utterance_length  = parser.getint('model', 'longest_utterance_length')
        self.use_delex_features        = parser.getboolean('model', 'delex_features')
        self.value_specific_decoder    = parser.getboolean('model', 'value_specific_decoder')
        self.learn_belief_state_update = parser.getboolean('model', 'learn_belief_state_update')

        print "  ---use_delex_features:", self.use_delex_features, \
            "\n  ---value_specific_decoder:", self.value_specific_decoder, \
            "\n  ---learn_belief_state_update", self.learn_belief_state_update

        self.reader = DataReader(self.wvecfile, self.ontologyfile, self.language,
                                 self.trainfile, self.testfile, self.validfile, self.percentage)

        self.dialogue_ontology = self.reader.dialogue_ontology
        self.word_vectors = self.reader.word_vectors
        self.word_vector_size = self.reader.word_vector_size
        self.data = self.reader.data

        # Neural Net Initialisation (keep variables packed so we can move them to either method):
        self.model_variables = {}
        self._init_model()

        # for track, avoid reload model
        self.sessions = {}

    def _init_model(self):
        """
        This method initialize neural belief tracker model variables for every informable slot and request.
        """
        for slot in self.dialogue_ontology:
            print "========  Initialisation of model variables for slot:", slot, "  ========"
            if slot == "request":
                slot_vectors  = numpy.zeros((len(self.dialogue_ontology[slot]), self.word_vector_size), dtype="float32")
                value_vectors = numpy.zeros((len(self.dialogue_ontology[slot]), self.word_vector_size), dtype="float32")

                for value_idx, value in enumerate(self.dialogue_ontology[slot]):
                    slot_vectors[value_idx, :] = self.word_vectors[slot]
                    value_vectors[value_idx, :] = self.word_vectors[value]

                self.model_variables[slot] = NeuralBeliefTracker(
                    self.word_vector_size, len(self.dialogue_ontology[slot]),
                    slot_vectors, value_vectors, self.longest_utterance_length,
                    use_delex_features=self.use_delex_features,
                    use_softmax=False,
                    value_specific_decoder=self.value_specific_decoder,
                    learn_belief_state_update=self.learn_belief_state_update)
            else:
                slot_vectors  = numpy.zeros((len(self.dialogue_ontology[slot]) + 1, self.word_vector_size), dtype="float32")  # +1 for None
                value_vectors = numpy.zeros((len(self.dialogue_ontology[slot]) + 1, self.word_vector_size), dtype="float32")

                for value_idx, value in enumerate(self.dialogue_ontology[slot]):
                    slot_vectors[value_idx, :] = self.word_vectors[slot]
                    value_vectors[value_idx, :] = self.word_vectors[value]

                self.model_variables[slot] = NeuralBeliefTracker(
                    self.word_vector_size, len(self.dialogue_ontology[slot]),
                    slot_vectors, value_vectors, self.longest_utterance_length,
                    use_delex_features=self.use_delex_features,
                    use_softmax=True,
                    value_specific_decoder=self.value_specific_decoder,
                    learn_belief_state_update=self.learn_belief_state_update)

    def train_net(self):
        """
        This method trains a model on the data and saves the file parameters to a file which can
        then be loaded to do evaluation.
        """
        # setting train and valid utterances
        _, utterances_train2 = self.data['train']
        _, utterances_valid2 = self.data['valid']
        valid_count2 = len(utterances_valid2)

        # TODO:: Change train/valid ratio.
        train_of_valid_ratio = 0.75
        utterances_train = utterances_train2 + utterances_valid2[: int(train_of_valid_ratio * valid_count2)]
        utterances_valid = utterances_valid2[int(train_of_valid_ratio * valid_count2):]
        print "  ---Training using:", self.trainfile, "count:", len(utterances_train)
        print "  ---Validate using:", self.validfile, "count:", len(utterances_valid)

        # training feature vectors and positive and negative examples list.
        print "\nGenerating data for training set and validation set:"
        feature_vectors_train, positive_examples_train, negative_examples_train = self._generate_data(utterances_train)
        feature_vectors_valid, positive_examples_valid, negative_examples_valid = self._generate_data(utterances_valid)

        slots = self.dialogue_ontology.keys()
        ratio = {}  # random positive count
        for slot in slots:
            ratio[slot] = self.batch_size / 2   # positive : negative = 1 : 1, positive + negative = batch_size

        print "Doing", self.batches_per_epoch, "randomly drawn batches of size", self.batch_size, \
            "for", self.max_epoch, "training epochs.\n"

        for slot in slots:
            print "\n============  Training the NBT Model for slot", slot, "===========\n"
            start_time = time.time()

            keep_prob, x_full, x_delex, \
            requested_slots, system_act_confirm_slots, system_act_confirm_values, \
            y_, y_past_state, accuracy, \
            f_score, precision, recall, num_true_positives, \
            num_positives, classified_positives, y, predictions, true_predictions, \
            correct_prediction, true_positives, train_step, update_coefficient = self.model_variables[slot]

            val_data = self._generate_examples(slot, feature_vectors_valid, positive_examples_valid, negative_examples_valid)
            if val_data is None:
                print "val data is none"
                break

            # will be used to save model parameters with best validation scores.
            saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            # TODO:: Changed validation metric from accuracy to f_score?
            # TODO:: Add early stopping?
            best_accuracy = -0.01
            epoch = 0
            max_epoch = self.max_epoch
            last_update = -1

            # 'request' slot will converge quickly.
            if slot in ['request']:
                max_epoch = 40

            start_time_train = time.time()

            while epoch < max_epoch:
                sys.stdout.flush()
                epoch += 1

                for batch_id in range(self.batches_per_epoch):
                    random_positive_count = ratio[slot]
                    random_negative_count = self.batch_size - random_positive_count
                    batch_data = self._generate_examples(slot, feature_vectors_train,
                                                         positive_examples_train, negative_examples_train,
                                                         random_positive_count, random_negative_count)
                    (batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values,
                     batch_delex, batch_ys, batch_ys_prev) = batch_data

                    [_, cf, cp, cr, ca] = sess.run([train_step, f_score, precision, recall, accuracy],
                                                   feed_dict={x_full: batch_xs_full,
                                                              x_delex: batch_delex,
                                                              requested_slots: batch_sys_req,
                                                              system_act_confirm_slots: batch_sys_conf_slots,
                                                              system_act_confirm_values: batch_sys_conf_values,
                                                              y_: batch_ys, y_past_state: batch_ys_prev,
                                                              keep_prob: 0.5})
                # ===============================  VALIDATION  ===============================================
                start_time_val = time.time()
                current_accuracy = self._evaluate_model(sess, self.model_variables[slot], val_data)
                print "Epoch", epoch, "/", max_epoch, "[Accuracy] =", current_accuracy, "for slot:", slot, \
                    "Eval took", round(time.time() - start_time_val, 2), "seconds. Last update:", last_update, \
                    "/", max_epoch, "Best accuracy:", best_accuracy

                # and if we got a new high score for validation f-score, we need to save the parameters:
                if current_accuracy > best_accuracy:
                    last_update = epoch
                    if epoch < 100:
                        if int(epoch * 1.5) > max_epoch:
                            max_epoch = int(epoch * 1.5)
                    else:
                        if int(epoch * 1.2) > max_epoch:
                            max_epoch = int(epoch * 1.2)
                    print "\n ======  New best validation metric:", round(current_accuracy, 4), \
                        " - saving these parameters. Epoch is:", epoch, "/", max_epoch, "======\n"
                    best_accuracy = current_accuracy
                    saver.save(sess, self.modeldir + slot)

                elif current_accuracy == best_accuracy:
                    if last_update + max_epoch * 0.2 < epoch:
                        last_update = epoch
                        saver.save(sess, self.modeldir + slot)
                        print '\n=====  Update stop too early, force save!!  =====\n'

                if epoch % 5 == 0 or epoch == max_epoch:
                    print "Epoch", max(epoch - 4, 1), "to", epoch, "took", round(time.time() - start_time_train, 2), "seconds."
                    start_time_train = time.time()
            print "The best parameters achieved a validation metric of", round(best_accuracy, 4)
            print "\n============  Training this model took", round(time.time() - start_time, 1), "seconds.  ============\n"

    def test_net(self):

        print "======  Testing NBT with language:", self.language, " ======"

        dialogues, training_turns = self.data['test']
        sessions = {}
        indexed_evaluated_dialogue = []
        list_of_belief_states = []

        # loading pre-trained models
        saver = tf.train.Saver()
        slots_to_load = ["food", "pricerange", "area", "request"]
        if self.language == "english":
            slots_to_load = ["food", "pricerange", "area", "request"]
        elif self.language == "italian":
            slots_to_load = ["cibo", "prezzo", "area", "request"]
        elif self.language == "german":
            slots_to_load = ["essen", "preisklasse", "gegend", "request"]
        for load_slot in slots_to_load:
            path_to_load = self.modeldir + load_slot
            print "------  Loading Model", path_to_load, " ------"
            sessions[load_slot] = tf.Session()
            saver.restore(sessions[load_slot], path_to_load)

        # evaluate
        dialogue_count = len(dialogues)
        for idx in range(dialogue_count):
            if idx % 100 == 0 or idx == dialogue_count - 1:
                print idx, "/", dialogue_count, "done."

            evaluated_dialogue, belief_states = self._track_dialogue(dialogues[idx], sessions)
            indexed_evaluated_dialogue.append({"dialogue_idx": idx, "dialogue": evaluated_dialogue})
            list_of_belief_states.append(belief_states)

        results = self._evaluate_dialogues(indexed_evaluated_dialogue)

        json.dump(indexed_evaluated_dialogue, open(self.resultfile, "w"), indent=4)
        print json.dumps(results, indent=4)

    def track_utterance(self, asr_obs, sysreq, sysconf_slot, sysconf_value, prev_belief_state):
        asr_obs = sorted(asr_obs, key=lambda asr_ob: asr_ob[1], reverse=True)
        utterance = normalize_transcription(asr_obs[0][0], self.language)
        cur_asr = [(normalize_transcription(hyp, self.language), score) for (hyp, score) in asr_obs]

        if not prev_belief_state:
            prev_belief_state = {"food": "none", "area": "none", "pricerange": "none"}

        utterances = [((utterance, cur_asr), list(sysreq), list(sysconf_slot), list(sysconf_value), prev_belief_state)]

        slots = self.dialogue_ontology.keys()

        if not self.sessions:
            saver = tf.train.Saver()
            for slot in slots:
                try:
                    path_to_load = self.modeldir + slot
                    self.sessions[slot] = tf.Session()
                    saver.restore(self.sessions[slot], path_to_load)
                except:
                    print "Can't restore for slot", slot, " - from file", path_to_load
                    return

        prediction_dict = {}
        distribution_dict = {}
        distribution_value_dict = {}

        for slot in slots:

            distribution = self._test_utterance(utterances, self.sessions[slot], self.model_variables[slot], slot)[0]

            distribution_dict[slot] = list(distribution)
            distribution_value_dict[slot] = {}
            values = self.dialogue_ontology[slot]
            for idx, value in enumerate(values):
                distribution_value_dict[slot][value] = distribution[idx]
            if slot != "request":
                distribution_value_dict[slot]["none"] = distribution[-1]

            if slot == "request":
                prediction_dict[slot] = self._value_of_belief_state_requestable(self.dialogue_ontology[slot],
                                                                                distribution, threshold=0.5)
            else:
                prediction_dict[slot] = self._value_of_belief_state_informable(self.dialogue_ontology[slot],
                                                                               distribution, threshold=0.01)
        return prediction_dict, distribution_value_dict, distribution_dict

    def _extract_feature_vectors(self, utterances, use_asr=False):
        """
        This method returns feature vectors for all dialogue utterances.
        It returns a tuple of lists, where each list consists of all feature vectors for ngrams of that length.
        This method doesn't care about the labels: other methods assign actual or fake labels later on.
        This can run on any size, including a single utterance.
        """
        list_of_features = []
        for idx, utterance in enumerate(utterances):
            # ===========  extract ngram feature vectors  ====================
            if use_asr:
                full_asr = utterance[0][1]  # just use ASR
            else:
                full_asr = [(utterance[0][0], 1.0)]  # create (transcription, 1.0)
            asr_feature_vectors = []
            for (c_example, asr_coeff) in full_asr:
                full_fv = numpy.zeros((self.longest_utterance_length * self.word_vector_size,), dtype="float32")
                if c_example != "":
                    words_utterance = c_example.split()
                    for word_idx, word in enumerate(words_utterance):
                        word = unicode(word)
                        if word not in self.word_vectors:
                            self.word_vectors[word] = xavier_vector(word)
                        try:
                            full_fv[word_idx * self.word_vector_size :
                                    (word_idx + 1) * self.word_vector_size] = self.word_vectors[word]
                        except:
                            print "Something off with word:", word, word in self.word_vectors
                asr_feature_vectors.append(
                    numpy.reshape(full_fv, (self.longest_utterance_length, self.word_vector_size)))
            # list of [asr_count, 40, 300] into [asr_count * 40, 300]
            ngram_feature_vector = numpy.concatenate(asr_feature_vectors, axis=0)

            # TODO:: Change ngram_feature_vector from [asr_count * 40, 300] to [40, 300]

            # ============  extract system request vectors  ===================
            cur_sysreq = utterance[1]
            cur_sysreq_vector = numpy.zeros((self.word_vector_size,), dtype="float32")
            for req in cur_sysreq:
                if req != "":
                    cur_sysreq_vector += self.word_vectors[unicode(req)]

            # ===========  extract system confirm slot/value vectors  =========
            cur_sysconf_slot = utterance[2]
            cur_sysconf_value = utterance[3]
            cur_sysconf_slot_vector = numpy.zeros((self.word_vector_size,), dtype="float32")
            cur_sysconf_value_vector = numpy.zeros((self.word_vector_size,), dtype="float32")
            cur_sysconf_count = len(cur_sysconf_slot)
            for sub_idx in range(cur_sysconf_count):
                c_slot = cur_sysconf_slot[sub_idx]
                c_value = cur_sysconf_value[sub_idx]
                if c_slot != "" and c_value != "":
                    if " " not in c_slot:
                        cur_sysconf_slot_vector += self.word_vectors[unicode(c_slot)]
                    else:
                        constituent_words = c_slot.split()
                        for word in constituent_words:
                            cur_sysconf_slot_vector += self.word_vectors[unicode(word)]
                    if " " not in c_value:
                        cur_sysconf_value_vector += self.word_vectors[unicode(c_value)]
                    else:
                        constituent_words = c_value.split()
                        for word in constituent_words:
                            cur_sysconf_value_vector += self.word_vectors[unicode(word)]

            list_of_features.append((ngram_feature_vector,
                                     cur_sysreq_vector,
                                     cur_sysconf_slot_vector,
                                     cur_sysconf_value_vector))
        return list_of_features

    def _generate_data(self, utterances):
        """
        Generates a data representation we can subsequently use.
        Let's say negative requests are now - those utterances which express no requestables.
        """
        # each list element is a tuple with features for that utterance
        # i.e. list of (transcription_vectors, sysreq_vector, sysconf_slot_vector, sysconf_value_vector)
        feature_vectors = self._extract_feature_vectors(utterances)

        # indexed by slot, these two dictionaries contain lists of positive and negative examples
        # for training each slot. Each list element is (utterance_id, utterance, value_id)
        positive_examples = {}
        negative_examples = {}

        slots = self.dialogue_ontology.keys()  # {'request', 'food', 'area', 'pricerange'}
        for slot_idx, slot in enumerate(slots):
            positive_examples[slot] = []
            negative_examples[slot] = []

            for utterance_idx, utterance in enumerate(utterances):
                slot_expressed_in_utterance = False

                # utterance[4] is the current belief state
                # utterance[5] is the previous belief state
                # utterance[6] is the current turn label
                # TODO:: Change positive/negative labels from current_turn_label to current_belief_state ?
                for (c_slot, c_value) in utterance[4]:
                    if c_slot == slot and (c_value != "none" and c_value != []):
                        slot_expressed_in_utterance = True  # if this is True, no negative examples for softmax.
                if slot != "request":
                    # if utterance is positive example for one value, it will not be negative example for others.
                    # if utterance is not expressed, it will be negative example for all values.
                    for value_idx, value in enumerate(self.dialogue_ontology[slot]):
                        if (slot, value) in utterance[4]:
                            positive_examples[slot].append((utterance_idx, utterance, value_idx))
                        else:
                            if not slot_expressed_in_utterance:
                                negative_examples[slot].append((utterance_idx, utterance, value_idx))
                else:
                    if not slot_expressed_in_utterance:
                        negative_examples[slot].append((utterance_idx, utterance, []))
                    else:
                        values_expressed = []
                        for value_idx, value in enumerate(self.dialogue_ontology[slot]):
                            if (slot, value) in utterance[4]:
                                values_expressed.append(value_idx)
                        positive_examples[slot].append((utterance_idx, utterance, values_expressed))

        return feature_vectors, positive_examples, negative_examples

    def _request_one_hot(self, value_idx):
        """
        takes a list, i.e. 2,3,4, and if req count is 8, returns: 00111000
        """
        request_count = len(self.dialogue_ontology["request"])
        zeros = numpy.zeros((request_count,), dtype=numpy.float32)
        for idx in value_idx:
            zeros[idx] = 1.0
        return zeros

    def _delexicalise_utterance_values(self, utterance, target_slot):
        """
        Takes a list of words which represent the current utterance, the loaded vectors, finds all occurrences of both slot name and slot value,
        and then returns the updated vector with "delexicalised tag" in them.
        """
        if type(utterance) is list:
            utterance = " ".join(utterance)
        target_values = self.dialogue_ontology[target_slot]
        if target_slot == "request":
            value_count = len(target_values)
        else:
            value_count = len(target_values) + 1    # NONE
        delexicalised_vector = numpy.zeros((value_count,), dtype="float32")
        for idx, value in enumerate(target_values):
            if " " + value + " " in utterance:
                delexicalised_vector[idx] = 1.0
        return delexicalised_vector

    def _generate_examples(self, target_slot, feature_vectors,
                           positive_examples, negative_examples,
                           positive_count=None, negative_count=None):
        """
        This method returns a minibatch of positive_count examples followed by negative_count examples.
        If these two are not set, it creates the full dataset (used for validation. if set, used for test).
        It returns: (features_transcription, features_system_request,
                     features_system_confirm_slot, features_system_confirm_values,
                     features_delex, features_previous_belief_state) - all we need to pass to train.
        """
        examples = []
        labels = []
        fv_full = []
        fv_sysreq = []
        fv_sysconf_slot = []
        fv_sysconf_value = []
        fv_delex = []
        fv_prev_bs = []

        # ===========  Choose positive_count + negative_count examples from all  ============
        pos_example_count = len(positive_examples[target_slot])
        neg_example_count = len(negative_examples[target_slot])
        if positive_count is None:
            positive_count = pos_example_count
        if negative_count is None:
            negative_count = neg_example_count
        if pos_example_count == 0 or positive_count == 0 or neg_example_count == 0 or negative_count == 0:
            print "#### SKIPPING (NO DATA): ", target_slot, \
                pos_example_count, positive_count, neg_example_count, negative_count
            return None

        positive_indices = []
        negative_indices = []
        if positive_count > 0:
            positive_indices = numpy.random.choice(pos_example_count, positive_count)
        if negative_count > 0:
            negative_indices = numpy.random.choice(neg_example_count, negative_count)
        for idx in positive_indices:
            examples.append(positive_examples[target_slot][idx])
        for idx in negative_indices:
            examples.append(negative_examples[target_slot][idx])

        # ==========================  Go through all examples  ==============================
        # feature_vectors i.e. list of (ngram_feature_vector, sysreq_vector, sysconf_slot_vector, sysconf_value_vector)
        # examples i.e. list of (utterance_id, utterance, value_id)
        # utterance i.e. ((trans, asr), sysreq, sysconf_slot, sysconf_value, cur_bs, prev_bs)
        if target_slot != "request":
            label_size = len(self.dialogue_ontology[target_slot]) + 1  # NONE
        else:
            label_size = len(self.dialogue_ontology[target_slot])
        for idx_example, example in enumerate(examples):
            (utterance_idx, utterance, value_idx) = example
            utterance_fv = feature_vectors[utterance_idx]

            if idx_example < positive_count:    # positive example
                if target_slot != "request":
                    labels.append(value_idx)    # includes dontcare
                else:
                    labels.append(self._request_one_hot(value_idx))
            else:                               # negative example
                if target_slot != "request":
                    labels.append(label_size - 1)  # NONE
                else:
                    labels.append([])           # wont ever use this

            # for now, we just deal with the utterance, and not with WOZ data.
            # TODO:: need to get a series of delexicalised vectors, one for each value.
            delex_feature_vector = self._delexicalise_utterance_values(utterance[0][0], target_slot)

            prev_belief_state = utterance[5]  # prev_belief_state is in utterance[5]
            prev_belief_state_vector = numpy.zeros((label_size,), dtype="float32")
            if target_slot != "request":
                prev_value = prev_belief_state[target_slot]
                if prev_value == "none" or prev_value not in self.dialogue_ontology[target_slot]:
                    prev_belief_state_vector[label_size - 1] = 1
                else:
                    prev_belief_state_vector[self.dialogue_ontology[target_slot].index(prev_value)] = 1

            fv_full.append(utterance_fv[0])
            fv_sysreq.append(utterance_fv[1])
            fv_sysconf_slot.append(utterance_fv[2])
            fv_sysconf_value.append(utterance_fv[3])
            fv_delex.append(delex_feature_vector)
            fv_prev_bs.append(prev_belief_state_vector)

        fv_full = numpy.array(fv_full)
        fv_sysreq = numpy.array(fv_sysreq)
        fv_sysconf_slot = numpy.array(fv_sysconf_slot)
        fv_sysconf_value = numpy.array(fv_sysconf_value)
        fv_delex = numpy.array(fv_delex)
        fv_prev_bs = numpy.array(fv_prev_bs)

        y_labels = numpy.zeros((positive_count + negative_count, label_size), dtype="float32")

        for idx in range(positive_count):
            if target_slot != "request":
                y_labels[idx, labels[idx]] = 1
            else:
                y_labels[idx, :] = labels[idx]

        if target_slot != "request":
            y_labels[positive_count:, label_size - 1] = 1  # None

        return fv_full, fv_sysreq, fv_sysconf_slot, fv_sysconf_value, fv_delex, y_labels, fv_prev_bs

    def _evaluate_model(self, sess, model_variables, data):

        keep_prob, x_full, x_delex, \
        requested_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, accuracy, \
        f_score, precision, recall, num_true_positives, \
        num_positives, classified_positives, y, predictions, true_predictions, correct_prediction, \
        true_positives, train_step, update_coefficient = model_variables

        (xs_full, xs_sys_req, xs_conf_slots, xs_conf_values, xs_delex, xs_labels, xs_prev_labels) = data

        example_count = xs_full.shape[0]
        label_size = xs_labels.shape[1]
        batch_size = 16
        batch_count = int(math.ceil(float(example_count) / batch_size))
        total_accuracy = 0.0
        element_count = 0

        for idx in range(batch_count):
            left_range = idx * batch_size
            right_range = min((idx+1)*batch_size, example_count)
            curr_len = right_range - left_range  # in the last batch, could be smaller than batch size

            if idx in [batch_count - 1, 0]:
                xss_full = numpy.zeros((batch_size, self.longest_utterance_length, self.word_vector_size), dtype="float32")
                xss_sys_req = numpy.zeros((batch_size, self.word_vector_size), dtype="float32")
                xss_conf_slots = numpy.zeros((batch_size, self.word_vector_size), dtype="float32")
                xss_conf_values = numpy.zeros((batch_size, self.word_vector_size), dtype="float32")
                xss_delex = numpy.zeros((batch_size, label_size), dtype="float32")
                xss_labels = numpy.zeros((batch_size, label_size), dtype="float32")
                xss_prev_labels = numpy.zeros((batch_size, label_size), dtype="float32")

            xss_full[0:curr_len, :, :] = xs_full[left_range:right_range, :, :]
            xss_sys_req[0:curr_len, :] = xs_sys_req[left_range:right_range, :]
            xss_conf_slots[0:curr_len, :] = xs_conf_slots[left_range:right_range, :]
            xss_conf_values[0:curr_len, :] = xs_conf_values[left_range:right_range, :]
            xss_delex[0:curr_len, :] = xs_delex[left_range:right_range, :]
            xss_labels[0:curr_len, :] = xs_labels[left_range:right_range, :]
            xss_prev_labels[0:curr_len, :] = xs_prev_labels[left_range:right_range, :]

            [current_predictions, current_y, current_accuracy, update_coefficient_load] = sess.run(
                [predictions, y, accuracy, update_coefficient],
                feed_dict={x_full: xss_full, x_delex: xss_delex,
                           requested_slots: xss_sys_req, system_act_confirm_slots: xss_conf_slots,
                           system_act_confirm_values: xss_conf_values, y_: xss_labels, y_past_state: xss_prev_labels,
                           keep_prob: 1.0})

            total_accuracy += current_accuracy
            element_count += 1

        eval_accuracy = round(total_accuracy / element_count, 3)

        return eval_accuracy

    def _test_utterance(self, utterances, sess, model_variables, target_slot):
        """
        Returns a list of belief states, to be weighted later.
        """
        list_of_belief_states = []

        # setting value count (add none to informable slot)
        value_count = len(self.dialogue_ontology[target_slot])
        if target_slot != "request":
            value_count = value_count + 1     # None

        # =====================================  TESTING  ==========================================
        # utterances i.e.list of((trans, asr), sysreq, sysconf_slot, sysconf_value, cur_belief_state, prev_belief_state)
        # feature_vectors i.e. list of (ngram_feature_vector, sysreq_vector, sysconf_slot_vector, sysconf_value_vector)
        fv_full = []
        fv_sysreq = []
        fv_sysconf_slot = []
        fv_sysconf_value = []
        fv_delex = []
        fv_prev_bs = []
        feature_vectors = self._extract_feature_vectors(utterances, use_asr=True)
        for idx_utterance, utterance_fv in enumerate(feature_vectors):
            prev_belief_state = utterances[idx_utterance][4]
            prev_belief_state_vector = numpy.zeros((value_count,), dtype="float32")
            if target_slot != "request":
                if isinstance(prev_belief_state[target_slot], list) and \
                        len(prev_belief_state[target_slot]) == value_count:
                    prev_belief_state_vector = numpy.array(prev_belief_state[target_slot], dtype="float32")
                else:
                    if type(prev_belief_state[target_slot]) in [str, unicode]:
                        prev_value = prev_belief_state[target_slot]
                        if prev_value == "none" or prev_value not in self.dialogue_ontology[target_slot]:
                            prev_belief_state_vector[value_count - 1] = 1   # None
                        else:
                            prev_belief_state_vector[self.dialogue_ontology[target_slot].index(prev_value)] = 1
                    elif type(prev_belief_state[target_slot]) in [dict]:
                        values = prev_belief_state[target_slot].keys()
                        for val in values:
                            if val == "none":
                                prev_belief_state_vector[-1] = prev_belief_state[target_slot][val]
                            else:
                                prev_belief_state_vector[self.dialogue_ontology[target_slot].index(val)] = \
                                    prev_belief_state[target_slot][val]

            delex_feature_vector = self._delexicalise_utterance_values(utterances[idx_utterance][0][0], target_slot)

            fv_full.append(utterance_fv[0])
            fv_sysreq.append(utterance_fv[1])
            fv_sysconf_slot.append(utterance_fv[2])
            fv_sysconf_value.append(utterance_fv[3])
            fv_delex.append(delex_feature_vector)
            fv_prev_bs.append(prev_belief_state_vector)

        fv_full = numpy.array(fv_full)
        fv_sysreq = numpy.array(fv_sysreq)
        fv_sysconf_slot = numpy.array(fv_sysconf_slot)
        fv_sysconf_value = numpy.array(fv_sysconf_value)
        fv_delex = numpy.array(fv_delex)
        fv_prev_bs = numpy.array(fv_prev_bs)

        keep_prob, x_full, x_delex, \
        requested_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, accuracy, \
        f_score, precision, recall, num_true_positives, \
        num_positives, classified_positives, y, predictions, true_predictions, correct_prediction, \
        true_positives, train_step, update_coefficient = model_variables

        distribution, update_coefficient_load = sess.run([y, update_coefficient],
                                                         feed_dict={x_full: fv_full, x_delex: fv_delex,
                                                                    requested_slots: fv_sysreq,
                                                                    system_act_confirm_slots: fv_sysconf_slot,
                                                                    system_act_confirm_values: fv_sysconf_value,
                                                                    y_past_state: fv_prev_bs,
                                                                    keep_prob: 1.0})

        utterance_count = len(utterances)
        for idx in range(utterance_count):
            cur_distribution = distribution[idx, :]
            list_of_belief_states.append(cur_distribution)

        return list_of_belief_states

    def _value_of_belief_state_requestable(self, values, distribution, threshold=0.5):
        """
        Returns the list of values which above threshold.
        """
        sysreq = []
        for idx, value in enumerate(values):
            if distribution[idx] >= threshold:
                sysreq.append(value)
        return sysreq

    def _value_of_belief_state_informable(self, values, distribution, threshold=0.5):
        """
        Returns the top one if it is above threshold.
        """
        max_value = "none"
        max_score = 0.0
        total_value = 0.0

        for idx, value in enumerate(values):
            total_value += distribution[idx]
            if distribution[idx] >= threshold:
                if distribution[idx] >= max_score:
                    max_value = value
                    max_score = distribution[idx]
        if max_score >= (1.0 - total_value):
            return max_value
        else:
            return "none"

    def _compare_request_lists(self, list_a, list_b):
        if len(list_a) != len(list_b):
            return False
        list_a.sort()
        list_b.sort()
        list_length = len(list_a)
        for idx in range(list_length):
            if list_a[idx] != list_b[idx]:
                return False
        return True

    def _track_dialogue(self, dialogue, sessions):
        """
        This method produces a list of belief states predicted for the given dialogue.
        """
        # to be able to combine predictions, we must also return the belief states for each turn.
        # So for each turn, a dictionary indexed by slot values which points to the distribution.
        predictions_for_dialogue = []
        list_of_predicted_belief_states = []

        # dialogue i.e. list of ((trans, asr), sysreq, sysconf_slot, sysconf_value, cur_belief_state, prev_belief_state)
        slots_to_track = list(set(self.dialogue_ontology.keys()) & set(sessions.keys()))
        for idx, utterance in enumerate(dialogue):
            list_of_predicted_belief_states.append({})
            predicted_bs = {}

            for slot in slots_to_track:
                # We will predict current_belief_state, so the true current belief state we set empty.
                if idx == 0 or slot == "request":
                    # this prev_belief_state should be empty
                    example = [(utterance[0], utterance[1], utterance[2], utterance[3], utterance[5])]
                else:
                    # and this has the previous prediction, the one we just made in the previous iteration.
                    # We do not want to use the right one, the one used for training.
                    example = [(utterance[0], utterance[1], utterance[2], utterance[3], prev_bs)]

                updated_belief_state = self._test_utterance(example, sessions[slot], self.model_variables[slot], slot)[0]
                list_of_predicted_belief_states[idx][slot] = list(updated_belief_state)

                if slot == "request":
                    predicted_bs[slot] = self._value_of_belief_state_requestable(self.dialogue_ontology[slot],
                                                                                 updated_belief_state, threshold=0.5)
                else:
                    predicted_bs[slot] = self._value_of_belief_state_informable(self.dialogue_ontology[slot],
                                                                                updated_belief_state, threshold=0.01)
            prev_bs = deepcopy(list_of_predicted_belief_states[idx])

            trans_plus_sys = "User: " + str(utterance[0][0])
            trans_plus_sys += "  ASR: " + str(utterance[0][1])
            if utterance[1][0] != "":
                trans_plus_sys += "  System Request: " + str(utterance[1])
            if utterance[2][0] != "":
                trans_plus_sys += "  System Confirm: " + str(utterance[2]) + ":" + str(utterance[3])

            predictions_for_dialogue.append((trans_plus_sys, {"True State": utterance[4]}, {"Prediction": predicted_bs}))

        return predictions_for_dialogue, list_of_predicted_belief_states

    def _evaluate_dialogues(self, indexed_evaluated_dialogue):
        """
        Given a list of {"dialogue_idx": idx, "dialogue": evaluated_dialogue}
            evaluated_dialogue i.e. (trans_plus_sys, correct labels, predicted labels)
        this measures joint goal (as in Matt's paper),
        and f-scores, as presented in Shawn's NIPS paper.

        Assumes request is always there in the ontology.
        """
        informable_slots = []
        if self.language == "english" or self.language == "en":
            informable_slots = ["food", "pricerange", "area"]
        elif self.language == "italian" or self.language == "it":
            informable_slots = ["cibo", "prezzo", "area"]
        elif self.language == "german" or self.language == "de":
            informable_slots = ["essen", "preisklasse", "gegend"]
        informable_slots = list(set(informable_slots) & set(self.dialogue_ontology.keys()))

        req_slots = []
        requestables = []
        if "request" in self.dialogue_ontology:
            req_slots = [str("req_"+x) for x in self.dialogue_ontology["request"]]
            requestables = ["request"]

        true_positives = {}
        false_negatives = {}
        false_positives = {}

        req_match = 0.0
        req_full_turn_count = 0.0

        req_acc_total = 0.0     # number of turns which express requestables
        req_acc_correct = 0.0

        for slot in self.dialogue_ontology:
            true_positives[slot] = 0
            false_positives[slot] = 0
            false_negatives[slot] = 0

        for value in requestables + req_slots + ["request"]:
            true_positives[value] = 0
            false_positives[value] = 0
            false_negatives[value] = 0

        correct_turns = 0       # when there is at least one informable, do all of them match?
        incorrect_turns = 0     # when there is at least one informable, if any are different.

        slot_correct_turns = {}
        slot_incorrect_turns = {}
        for slot in informable_slots:
            slot_correct_turns[slot] = 0.0
            slot_incorrect_turns[slot] = 0.0

        dialogue_joint_metrics = []
        dialogue_req_metrics = []

        dialogue_slot_metrics = {}
        for slot in informable_slots:
            dialogue_slot_metrics[slot] = []

        dialogue_count = len(indexed_evaluated_dialogue)
        for idx in range(dialogue_count):
            dialogue = indexed_evaluated_dialogue[idx]["dialogue"]

            curr_dialogue_goal_joint_total = 0.0    # how many turns have informables
            curr_dialogue_goal_joint_correct = 0.0

            curr_dialogue_goal_slot_total = {}      # how many turns in current dialogue have specific informable
            curr_dialogue_goal_slot_correct = {}    # and how many of these are correct

            for slot in informable_slots:
                curr_dialogue_goal_slot_total[slot] = 0.0
                curr_dialogue_goal_slot_correct[slot] = 0.0

            creq_tp = 0.0
            creq_fn = 0.0
            creq_fp = 0.0

            # to compute per-dialogue f-score for requestables
            for turn in dialogue:
                # first update full requestable
                req_full_turn_count += 1.0

                if requestables:

                    if self._compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                        req_match += 1.0

                    if len(turn[1]["True State"]["request"]) > 0:
                        req_acc_total += 1.0

                        if self._compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                            req_acc_correct += 1.0

                # per dialogue requestable metrics
                if requestables:

                    true_requestables = turn[1]["True State"]["request"]
                    predicted_requestables = turn[2]["Prediction"]["request"]

                    for each_true_req in true_requestables:
                        if each_true_req in self.dialogue_ontology["request"] and each_true_req in predicted_requestables:
                            true_positives["request"] += 1
                            creq_tp += 1.0
                            true_positives["req_" + each_true_req] += 1
                        elif each_true_req in self.dialogue_ontology["request"]:
                            false_negatives["request"] += 1
                            false_negatives["req_" + each_true_req] += 1
                            creq_fn += 1.0
                            # print "FN:", turn[0], "---", true_requestables, "----", predicted_requestables

                    for each_predicted_req in predicted_requestables:
                        # ignore matches, already counted, now need just negatives:
                        if each_predicted_req not in true_requestables:
                            false_positives["request"] += 1
                            false_positives["req_" + each_predicted_req] += 1
                            creq_fp += 1.0
                            # print "-- FP:", turn[0], "---", true_requestables, "----", predicted_requestables

                # print turn
                inf_present = {}
                inf_correct = {}

                for slot in informable_slots:
                    inf_present[slot] = False
                    inf_correct[slot] = True

                informable_present = False
                informable_correct = True

                for slot in informable_slots:

                    try:
                        true_value = turn[1]["True State"][slot]
                        predicted_value = turn[2]["Prediction"][slot]
                    except:

                        print "PROBLEM WITH", turn, "slot:", slot, "inf slots", informable_slots

                    if true_value != "none":
                        informable_present = True
                        inf_present[slot] = True

                    if true_value == predicted_value:  # either match or none, so not incorrect
                        if true_value != "none":
                            true_positives[slot] += 1
                    else:
                        if true_value == "none":
                            false_positives[slot] += 1
                        elif predicted_value == "none":
                            false_negatives[slot] += 1
                        else:
                            # spoke to Shawn - he does this as false negatives for now - need to think about how we evaluate it properly.
                            false_negatives[slot] += 1

                        informable_correct = False
                        inf_correct[slot] = False

                if informable_present:

                    curr_dialogue_goal_joint_total += 1.0

                    if informable_correct:
                        correct_turns += 1
                        curr_dialogue_goal_joint_correct += 1.0
                    else:
                        incorrect_turns += 1

                for slot in informable_slots:
                    if inf_present[slot]:
                        curr_dialogue_goal_slot_total[slot] += 1.0

                        if inf_correct[slot]:
                            slot_correct_turns[slot] += 1.0
                            curr_dialogue_goal_slot_correct[slot] += 1.0
                        else:
                            slot_incorrect_turns[slot] += 1.0

                # current dialogue requestables

            if creq_tp + creq_fp > 0.0:
                creq_precision = creq_tp / (creq_tp + creq_fp)
            else:
                creq_precision = 0.0

            if creq_tp + creq_fn > 0.0:
                creq_recall = creq_tp / (creq_tp + creq_fn)
            else:
                creq_recall = 0.0

            if creq_precision + creq_recall == 0:
                if creq_tp == 0 and creq_fn == 0 and creq_fn == 0:
                    # no requestables expressed, special value
                    creq_fscore = -1.0
                else:
                    creq_fscore = 0.0  # none correct but some exist
            else:
                creq_fscore = (2 * creq_precision * creq_recall) / (creq_precision + creq_recall)

            dialogue_req_metrics.append(creq_fscore)

            # and current dialogue informables:

            for slot in informable_slots:
                if curr_dialogue_goal_slot_total[slot] > 0:
                    dialogue_slot_metrics[slot].append(
                        float(curr_dialogue_goal_slot_correct[slot]) / curr_dialogue_goal_slot_total[slot])
                else:
                    dialogue_slot_metrics[slot].append(-1.0)

            if informable_slots:
                if curr_dialogue_goal_joint_total > 0:
                    current_dialogue_joint_metric = float(
                        curr_dialogue_goal_joint_correct) / curr_dialogue_goal_joint_total
                    dialogue_joint_metrics.append(current_dialogue_joint_metric)
                else:
                    # should not ever happen when all slots are used, but for validation we might not have
                    # i.e. area mentioned
                    dialogue_joint_metrics.append(-1.0)

        if informable_slots:
            goal_joint_total = float(correct_turns) / float(correct_turns + incorrect_turns)

        slot_gj = {}

        total_true_positives = 0
        total_false_negatives = 0
        total_false_positives = 0

        precision = {}
        recall = {}
        fscore = {}

        # FSCORE for each requestable slot:
        if requestables:
            add_req = ["request"] + req_slots
        else:
            add_req = []

        for slot in informable_slots + add_req:

            if slot not in ["request"] and slot not in req_slots:
                total_true_positives += true_positives[slot]
                total_false_positives += false_positives[slot]
                total_false_negatives += false_negatives[slot]

            precision_denominator = (true_positives[slot] + false_positives[slot])

            if precision_denominator != 0:
                precision[slot] = float(true_positives[slot]) / precision_denominator
            else:
                precision[slot] = 0

            recall_denominator = (true_positives[slot] + false_negatives[slot])

            if recall_denominator != 0:
                recall[slot] = float(true_positives[slot]) / recall_denominator
            else:
                recall[slot] = 0

            if precision[slot] + recall[slot] != 0:
                fscore[slot] = (2 * precision[slot] * recall[slot]) / (precision[slot] + recall[slot])
                print "REQ - slot", slot, round(precision[slot], 3), round(recall[slot], 3), round(fscore[slot], 3)
            else:
                fscore[slot] = 0

            total_count_curr = true_positives[slot] + false_negatives[slot] + false_positives[slot]\

        if requestables:

            requested_accuracy_all = req_match / req_full_turn_count

            if req_acc_total != 0:
                requested_accuracy_exist = req_acc_correct / req_acc_total
            else:
                requested_accuracy_exist = 1.0

            slot_gj["request"] = round(requested_accuracy_exist, 3)

        for slot in informable_slots:
            slot_gj[slot] = round(
                float(slot_correct_turns[slot]) / float(slot_correct_turns[slot] + slot_incorrect_turns[slot]), 3)

        # NIKOLA TODO: will be useful for goal joint
        if len(informable_slots) == 3:
            slot_gj["joint"] = round(goal_joint_total, 3)

        return slot_gj

