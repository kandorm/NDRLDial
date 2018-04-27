import tensorflow as tf


def CNNEncoder(utterance_representations_full, num_filters=300, vector_dimension=300, longest_utterance_length=40):
    '''
    Better code for defining the CNN Encoder.
    '''
    filter_sizes = [1, 2, 3]
    hidden_representation = tf.zeros([num_filters], tf.float32)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        # Convolution Layer
        filter_shape = [filter_size, vector_dimension, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            tf.expand_dims(utterance_representations_full, -1),
            W,
            strides=[1, 1, 1, 1],
            padding='VALID')
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, longest_utterance_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        pooled_outputs.append(pooled)
        hidden_representation += tf.reshape(tf.concat(pooled, 3), [-1, num_filters])

    return hidden_representation
