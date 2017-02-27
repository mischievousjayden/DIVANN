import tensorflow as tf


class DIVANN:
    def __init__(self, num_features, num_hidden, num_classes=1, beta=1.0):
        self._num_features = num_features
        self._num_hidden = num_hidden
        self._num_classes = num_classes
        self._beta = beta

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
        return self._decode, self.get_response_rule(x)

    def get_response_rule(self, x):
        ssqerror = tf.pow(tf.sub(self._decode, x), 2)
        diversities = tf.exp(self._beta*tf.reduce_mean(tf.abs(self._pdiff(self._decode)), 0))
        fweights = tf.div(diversities, tf.reduce_sum(diversities))
        ssqerror = tf.reduce_sum(tf.mul(ssqerror, fweights), 2)
        ssqerror = tf.reshape(ssqerror, [-1])
        ssqerror = tf.div(1.0, ssqerror)
        ps = tf.div(ssqerror, tf.reduce_sum(ssqerror))
        return ps

    def _pdiff(self, m):
        result = list()
        for i in range(self._num_classes):
            for j in range(i+1,self._num_classes):
                result.append(m[i]-m[j])
        return result

