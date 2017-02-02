import tensorflow as tf

from data import Data

# parameters
num_set_inputs = 13
training_epochs = 100
display_step = 5

# network parameters
num_features = 3
num_hidden = 8

# response rule beta
beta = 0.8

# tf graph input
X = tf.placeholder("float", [None, num_features], name="x-input")

weights = {
    "encoder": tf.Variable(tf.random_normal([num_features, num_hidden]), name="encoder_weight"),
    "decoder1": tf.Variable(tf.random_normal([num_hidden, num_features]), name="decoder_weight1"),
    "decoder2": tf.Variable(tf.random_normal([num_hidden, num_features]), name="decoder_weight2")
}

biases = {
    "encoder": tf.Variable(tf.random_normal([num_hidden]), name="encoder_bias"),
    "decoder1": tf.Variable(tf.random_normal([num_features]), name="decoder_bias1"),
    "decoder2": tf.Variable(tf.random_normal([num_features]), name="decoder_bias2")
}

# building the encoder
def encode(x):
    output = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder"]), biases["encoder"]))
    return output

# diva decoder generates multi-dimention output
def decodeToMultiDim(x):
    output1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder1"]), biases["decoder1"]))
    output2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder2"]), biases["decoder2"]))
    return [output1, output2]

def responseRule(outputs, label, beta):
    o1 = tf.pow(outputs[0] - label, [2, 2, 2])
    o2 = tf.pow(outputs[1] - label, [2, 2, 2])
    ssqerror = [o1, o2]
    diversities = tf.exp(beta*(tf.abs(outputs[0] - outputs[1])))
    fweights = tf.nn.softmax(diversities)
    accuracy1 = 1.0 / tf.reduce_sum(tf.mul(o1, fweights))
    accuracy2 = 1.0 / tf.reduce_sum(tf.mul(o2, fweights))
    return tf.nn.softmax([accuracy1, accuracy2])

# def getPrediction(outputs, current_class):
#     return outputs[current_class]

# construct model
with tf.name_scope("encoder_layer") as scope:
    encoder_op = encode(X)

with tf.name_scope("decoder_layer") as scope:
    decoder_op = decodeToMultiDim(encoder_op)

# prediction
current_class = tf.placeholder(tf.int32)
y_pred = tf.cond(current_class > 0 , lambda: decoder_op[1], lambda: decoder_op[0])

# target
y_true = X

# accuracy
with tf.name_scope("accuracy") as scope:
    accuracy_vec = responseRule(decoder_op, X, beta)
    accuracy = tf.cond(current_class > 0 , lambda: accuracy_vec[1], lambda: accuracy_vec[0])
    accuracy_summ = tf.scalar_summary("accuracy", accuracy)

# define cost
with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    cost_summ = tf.scalar_summary("cost", cost)

# Minimize
with tf.name_scope("train") as scope:
    learning_rate = 0.01
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# histogram
w_encoder_hist = tf.histogram_summary("weight_encoder", weights["encoder"])
w_decoder1_hist = tf.histogram_summary("weight_decoder1", weights["decoder1"])
w_decoder2_hist = tf.histogram_summary("weight_decoder2", weights["decoder2"])

b_encoder_hist = tf.histogram_summary("bias_encoder", biases["encoder"])
b_decoder1_hist = tf.histogram_summary("bias_decoder1", biases["decoder1"])
b_decoder2_hist = tf.histogram_summary("bias_decoder2", biases["decoder2"])

y_hist = tf.histogram_summary("prediction", y_pred)

# initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.merge_all_summaries()

    for i in range(num_set_inputs):
        logfilename = "./logs/set{}".format(i)
        writer = tf.train.SummaryWriter(logfilename, sess.graph_def)
        sess.run(init)
        for epoch in range(training_epochs):
            for j, current_input in enumerate(Data.Stimuli[i]):
                _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X:[current_input], current_class:Data.Assignments[j]})
                if epoch % display_step == 0:
                    summary = sess.run(merged, feed_dict={X:[current_input], current_class:Data.Assignments[j]})
                    writer.add_summary(summary, epoch)
                    print("input_set: {}".format(i), "Epoch: {}".format(epoch+1), "cost: {:.9f}".format(c), "accuracy={:.9f}".format(a), "label={}".format(Data.Assignments[j]))
    print("Optimization Finished!")
