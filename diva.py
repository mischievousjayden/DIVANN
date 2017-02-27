import tensorflow as tf


class DIVANN:
    def __init__(self, num_features, num_hidden, num_classes=1):
        self._num_features = num_features
        self._num_hidden = num_hidden
        self._num_classes = num_classes

        self._encoder = {
            "weight": tf.Variable(tf.random_normal([num_features, num_hidden]), name="encoder_weight"),
            "bias": tf.Variable(tf.random_normal([num_hidden]), name="encoder_bias")
        }

        self._decoder = {
            "weights": tf.Variable(tf.random_normal([num_classes, num_hidden, num_features]), name="decoder_weights"),
            "biases": tf.Variable(tf.random_normal([num_classes, num_features]), name="decoder_bias1")
        }

        self._histogram = list()
        self._histogram.append(tf.histogram_summary("encoder_weight", self._encoder["weight"]))
        self._histogram.append(tf.histogram_summary("encoder_bias", self._encoder["bias"]))
        self._histogram.append(tf.histogram_summary("decoder_weights", self._decoder["weights"]))
        self._histogram.append(tf.histogram_summary("decoder_biases", self._decoder["biases"]))

    def run(self, x):
        with tf.name_scope("encode") as scope:
            self._encode = tf.nn.sigmoid(tf.add(tf.matmul(x, self._encoder["weight"]), self._encoder["bias"]))

        with tf.name_scope("decode") as scope:
            self._decode = list()
            for i in range(self._num_classes):
                self._decode.append(tf.nn.sigmoid(tf.add(tf.matmul(self._encode, self._decoder["weights"][i]), self._decoder["biases"][i])))
        return self._decode
        
def responseRule(outputs, label, beta):
    o1 = tf.pow(outputs[0] - label, [2, 2, 2])
    o2 = tf.pow(outputs[1] - label, [2, 2, 2])
    ssqerror = [o1, o2]
    diversities = tf.exp(beta*(tf.abs(outputs[0] - outputs[1])))
    fweights = tf.nn.softmax(diversities)
    accuracy1 = 1.0 / tf.reduce_sum(tf.mul(o1, fweights))
    accuracy2 = 1.0 / tf.reduce_sum(tf.mul(o2, fweights))
    return tf.nn.softmax([accuracy1, accuracy2])

