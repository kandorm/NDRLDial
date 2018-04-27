import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from NBT.nn.encoder import CNNEncoder

def NeuralBeliefTracker(vector_dimension, label_count,
                 slot_vectors, value_vectors,
                 longest_utterance_length,
                 use_delex_features=False, use_softmax=True,
                 value_specific_decoder=False, learn_belief_state_update=True):
    """
    This method defines the model and returns the required TensorFlow operations.

    slot_vectors, value_vectors should be of size [label_count + 2, 300].   # +2 for none and dontcare
    For None, we should just pass zero vectors for both.

    Then, replicate using these vectors the old NBT and then combine each value's (including NONE) into softmax.

    List of values learned by this model:

    1) h_utterance_representation, which uses a CNN to learn a representation of the utterance r.
    2) candidates_transform, which includes w_candidates and b_candidates, which transforms candidate values to vector c.
    3) w_joint_hidden_layer and b_joint_hidden_layer, which collapses the interaction of r and c to an intermediate vector.
    4) w_joint_presoftmax and b_joint_presoftmax, which collapse the intermediate layer to a single feature.
    5) sysreq_w_hidden_layer and sysreq_b_hidden_layer, which compute intermediate sysreq representation.
    6) TODO: sysreq_w_softmax and sysreq_b_softmax, which map this to final decision. -- currently not size independent.
    7) TODO: confirm_w1_hidden_layer, confirm_b1_hidden_layer, confirm_w1_softmax, confirm_b1_softmax: for confirmations. -- currently does not work.
    8) a_memory, b_memory, a_current, b_current: for the belief state updates, composed into matrix.

    If all of these are initialised and then supplied to each of the models, we could train them together (batch of each slot), and just save
    these variables, then at test time, just load them (as session even), and then initialise all of the models with them.
    """
    if use_softmax:
        label_size = label_count + 1    # None
    else:
        label_size = label_count

    # these are actual NN hyperparameters that we might want to tune at some point
    hidden_units_1 = 100
    print "Hidden layer size:", hidden_units_1, "Label Size:", label_size, "Use Softmax:", use_softmax, "Use Delex Features:", use_delex_features

    utterance_representations_full  = tf.placeholder(tf.float32, [None, longest_utterance_length, vector_dimension]) # full feature vector, which we want to convolve over
    utterance_representations_delex = tf.placeholder(tf.float32, [None, label_size])
    system_act_slots                = tf.placeholder(tf.float32, [None, vector_dimension])  # just slots, for requestable
    system_act_confirm_slots        = tf.placeholder(tf.float32, [None, vector_dimension])
    system_act_confirm_values       = tf.placeholder(tf.float32, [None, vector_dimension])

    # Initial (distributional) vectors, Needed for L2 regularisation
    W_slots = tf.constant(slot_vectors, name="W_init")
    W_values = tf.constant(value_vectors, name="W_init")

    # output label, i.e, True / False, 1-hot encoded
    y_           = tf.placeholder(tf.float32, [None, label_size])
    y_past_state = tf.placeholder(tf.float32, [None, label_size])

    # dropout placeholder, 0.5 for training, 1.0 for validation/testing
    keep_prob = tf.placeholder("float")

    # =================================================================
    # =================  Utterance Representation  ====================
    # =================================================================

    # filter needs to be shape: filter_height=[1, 2, 3], filter_width=300, in_channel=1, out_channel = num_filters
    # filter just dot products - in images these then overlap from different regions - we don't have that.
    num_filters = 300
    h_utterance_representation = CNNEncoder(utterance_representations_full, num_filters, vector_dimension,
                                            longest_utterance_length)

    # =================================================================
    # =================  Candidate Representation  ====================
    # =================================================================

    # multiply to get: [label_count, num_filters]
    w_candidates = tf.Variable(tf.random_normal([vector_dimension, num_filters]))
    b_candidates = tf.Variable(tf.zeros([num_filters]))
    candidates_transform = tf.nn.sigmoid(tf.matmul(W_values, w_candidates) + b_candidates)

    # =================================================================
    # ================  Semantic Decoding  ============================
    # =================================================================

    # Multiply the h_utterance representations [None, num_filters] with
    # each of candidates [label_count, num_filters]
    # to get [None, label_count, num_filters]
    list_of_value_contributions = []
    for value_idx in range(label_count):
        list_of_value_contributions.append(
            tf.multiply(h_utterance_representation, candidates_transform[value_idx, :]))
    h_utterance_representation_candidate_interaction = \
        tf.reshape(tf.transpose(tf.stack(list_of_value_contributions), [1, 0, 2]), [-1, num_filters])

    # multiply [None, label_count, num_filters] by [num_filters, hidden_units_1],
    # to get [None, label_count, hidden_units_1]
    w_joint_hidden_layer = tf.Variable(tf.random_normal([num_filters, hidden_units_1]))
    b_joint_hidden_layer = tf.Variable(tf.zeros([hidden_units_1]))

    hidden_layer_joint = tf.nn.sigmoid(tf.reshape(
        tf.matmul(h_utterance_representation_candidate_interaction,
                  w_joint_hidden_layer) + b_joint_hidden_layer,
        [-1, label_count, hidden_units_1]))
    hidden_layer_joint_with_dropout = tf.nn.dropout(hidden_layer_joint, keep_prob)

    # initialize parameters that go into a softmax,
    # i.e. mapping [None, label_count, hidden_units_1] -> [None, label_count]
    w_joint_presoftmax = tf.Variable(tf.random_normal([hidden_units_1, 1])) # collapse to 1
    b_joint_presoftmax = tf.Variable(tf.zeros([1]))

    y_presoftmax = tf.reshape(
        tf.matmul(
            tf.reshape(hidden_layer_joint_with_dropout, [-1, hidden_units_1]),
            w_joint_presoftmax) + b_joint_presoftmax,
        [-1, label_count])

    # ===============================================================
    # ====================  Context Modelling  ======================
    # ===============================================================

    sysreq_contributions  = [] # a list of contributions for each of the values
    confirm_contributions = [] # a list of contributions for each of the values

    # ================  NETWORK FOR SYSTEM REQUESTS  ================

    # is the current slot offered
    system_act_candidate_interaction = tf.multiply(W_slots[0, :], system_act_slots) # only multiply with slots for the request
    dot_product_sysreq = tf.reduce_mean(system_act_candidate_interaction, 1)
    decision_sysreq = tf.multiply(tf.expand_dims(dot_product_sysreq, 1), h_utterance_representation)

    # allow each value to learn to map different utterances to yes, Mainly dontcare
    sysreq_w_hidden_layer = tf.Variable(tf.random_normal([num_filters, hidden_units_1]))
    sysreq_b_hidden_layer = tf.Variable(tf.zeros([hidden_units_1]))
    for value_idx in range(label_count):
        sysreq_hidden_layer_1 = tf.nn.sigmoid(tf.matmul(decision_sysreq, sysreq_w_hidden_layer) + sysreq_b_hidden_layer)
        sysreq_hidden_layer_1_with_dropout = tf.nn.dropout(sysreq_hidden_layer_1, keep_prob)
        sysreq_w_softmax = tf.Variable(tf.random_normal([hidden_units_1, 1]))
        sysreq_b_softmax = tf.Variable(tf.zeros([1]))
        sysreq_contribution = tf.matmul(sysreq_hidden_layer_1_with_dropout, sysreq_w_softmax) + sysreq_b_softmax
        sysreq_contributions.append(sysreq_contribution)

    sysreq = tf.concat(sysreq_contributions, 1)  # [None, label_count]

    # ================  NETWORK FOR CONFIRMATIONS  ====================

    # here, we want to tie across all values, as it will get a different signal depending on whether both things match.
    confirm_w1_hidden_layer = tf.Variable(tf.random_normal([num_filters, hidden_units_1]))
    confirm_b1_hidden_layer = tf.Variable(tf.zeros([hidden_units_1]))

    confirm_w1_softmax = tf.Variable(tf.random_normal([hidden_units_1, 1]))
    confirm_b1_softmax = tf.Variable(tf.zeros([1]))

    for value_idx in range(label_count):
        sysconf_slot_candidate_interaction = tf.multiply(W_slots[0, :], system_act_confirm_slots)
        dot_product_sysconf_slot = tf.reduce_mean(sysconf_slot_candidate_interaction, 1)

        sysconf_value_candidate_interaction = tf.multiply(W_values[value_idx, :], system_act_confirm_values)
        dot_product_sysconf_value = tf.reduce_mean(sysconf_value_candidate_interaction, 1)

        dot_product_sysconf = tf.multiply(dot_product_sysconf_slot, dot_product_sysconf_value)

        decision_sysconf = tf.multiply(tf.expand_dims(dot_product_sysconf, 1), h_utterance_representation)

        confirm_hidden_layer_1 = tf.nn.sigmoid(tf.matmul(decision_sysconf, confirm_w1_hidden_layer) + confirm_b1_hidden_layer)
        confirm_hidden_layer_1_with_dropout = tf.nn.dropout(confirm_hidden_layer_1, keep_prob)

        confirm_contribution = tf.matmul(confirm_hidden_layer_1_with_dropout, confirm_w1_softmax) + confirm_b1_softmax
        confirm_contributions.append(confirm_contribution)

    sysconf = tf.concat(confirm_contributions, 1)

    if use_softmax:

        # [None, label_count] -> [None, label_count + 1]
        append_zeros_none = tf.zeros([tf.shape(y_presoftmax)[0], 1])
        y_presoftmax = tf.concat([y_presoftmax, append_zeros_none], 1)

        append_zeros = tf.zeros([tf.shape(y_presoftmax)[0], 1])
        sysreq = tf.concat([sysreq, append_zeros], 1)
        sysconf = tf.concat([sysconf, append_zeros], 1)

        y_presoftmax = y_presoftmax + sysconf + sysreq

    if use_delex_features:
        y_presoftmax = y_presoftmax + utterance_representations_delex

    # value-specific decoder:
    if value_specific_decoder and False:
        h_utterance_representation_for_full_softmax = CNNEncoder(utterance_representations_full, num_filters,
                                                                       vector_dimension, longest_utterance_length)

        h_utterance_dropout = tf.nn.dropout(h_utterance_representation_for_full_softmax, keep_prob)

        ss_w_hidden_layer = tf.Variable(tf.random_normal([num_filters, hidden_units_1]))
        ss_b_hidden_layer = tf.Variable(tf.zeros([hidden_units_1]))

        ss_hidden_layer_1 = tf.nn.relu(tf.matmul(h_utterance_dropout, ss_w_hidden_layer) + ss_b_hidden_layer)
        ss_hidden_layer_1_with_dropout = tf.nn.dropout(ss_hidden_layer_1, keep_prob)

        ss_w_softmax = tf.Variable(tf.random_normal([hidden_units_1, label_size]))
        ss_b_softmax = tf.Variable(tf.zeros([label_size]))

        ss_contribution = tf.matmul(ss_hidden_layer_1_with_dropout, ss_w_softmax) + ss_b_softmax

        y_presoftmax += ss_contribution

    # as we are returning always, can't be null
    update_coefficient = tf.constant(0.49)

    if use_softmax:
        if learn_belief_state_update:
            if value_specific_decoder:  # value-specific update
                update_coefficient = tf.constant(0.8)
                ss_W_memory = tf.Variable(tf.random_normal([label_size, label_size]))
                ss_W_current = tf.Variable(tf.random_normal([label_size, label_size]))
                y_combine = tf.matmul(y_past_state, ss_W_memory) + tf.matmul(y_presoftmax, ss_W_current)
            else:
                a_memory = tf.Variable(tf.random_normal([1, 1]))
                diag_memory = a_memory * tf.diag(tf.ones(label_size))

                b_memory = tf.Variable(tf.random_normal([1, 1]))
                non_diag_memory = tf.matrix_set_diag(b_memory * tf.ones([label_size, label_size]),
                                                     tf.zeros(label_size))

                W_memory = diag_memory + non_diag_memory

                a_current = tf.Variable(tf.random_normal([1, 1]))
                diag_current = a_current * tf.diag(tf.ones(label_size))

                b_current = tf.Variable(tf.random_normal([1, 1]))
                non_diag_current = tf.matrix_set_diag(b_current * tf.ones([label_size, label_size]),
                                                      tf.zeros(label_size))

                W_current = diag_current + non_diag_current

                y_combine = tf.matmul(y_past_state, W_memory) + tf.matmul(y_presoftmax, W_current)

            y = tf.nn.softmax(y_combine)
        else:
            update_coefficient = tf.Variable(0.5)  # this scales the contribution of the current turn.
            y_combine = update_coefficient * y_presoftmax + (1 - update_coefficient) * y_past_state
            y = tf.nn.softmax(y_combine)
    else:
        y = tf.nn.sigmoid(y_presoftmax)  # for requestables, we just have turn-level binary decisions

    # ======================== LOSS IS JUST CROSS ENTROPY ==========================================
    if use_softmax:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_combine, labels=y_)
    else:
        cross_entropy = tf.reduce_sum(tf.square(y - y_))

    # ============================= EVALUATION =====================================================

    if use_softmax:
        predictions = tf.cast(tf.argmax(y, 1), "float32")  # will have ones where positive
        true_predictions = tf.cast(tf.argmax(y_, 1), "float32")
        correct_prediction = tf.cast(tf.equal(predictions, true_predictions), "float")

        accuracy = tf.reduce_mean(correct_prediction)
        # this will count number of positives - they are marked with 1 in true_predictions
        num_positives = tf.reduce_sum(true_predictions)     # TP + FN
        # positives are indicated with ones.
        classified_positives = tf.reduce_sum(predictions)   # TP + FP
        # will have ones in all places where both are predicting positives
        true_positives = tf.multiply(predictions, true_predictions)
        # if indicators for positive of both are 1, then it is positive.
        num_true_positives = tf.reduce_sum(true_positives)  # TP

        recall = num_true_positives / num_positives
        precision = num_true_positives / classified_positives
        f_score = (2 * recall * precision) / (recall + precision)

    else:
        predictions = tf.cast(tf.round(y), "float32")  # will have ones where positive
        true_predictions = tf.cast(tf.round(y_), "float32")
        correct_prediction = tf.cast(tf.equal(predictions, true_predictions), "float")

        num_positives = tf.reduce_sum(true_predictions)

        classified_positives = tf.reduce_sum(predictions)
        true_positives = tf.multiply(predictions, true_predictions)
        num_true_positives = tf.reduce_sum(true_positives)
        recall = num_true_positives / num_positives
        precision = num_true_positives / classified_positives
        f_score = (2 * recall * precision) / (recall + precision)

        accuracy = tf.reduce_mean(correct_prediction)

    optimizer = tf.train.AdamOptimizer(0.001)
    train_step = optimizer.minimize(cross_entropy)

    return keep_prob, utterance_representations_full, utterance_representations_delex, \
           system_act_slots, system_act_confirm_slots, system_act_confirm_values, \
           y_, y_past_state, accuracy, f_score, precision, recall, \
           num_true_positives, num_positives, classified_positives, y, \
           predictions, true_predictions, correct_prediction, true_positives, train_step, update_coefficient
