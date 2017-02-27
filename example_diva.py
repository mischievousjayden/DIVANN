import tensorflow as tf

from data import Data
import diva
from diva import DIVANN


def main():
    # parameters
    num_set_inputs = 13
    training_epochs = 100
    display_step = 5

    # network parameters
    num_features = 3
    num_hidden = 8
    num_classes = 2
    beta = 0.8
    divann = DIVANN(num_features, num_hidden, num_classes, beta)

    # tf graph input
    X = tf.placeholder("float", [None, num_features], name="x-input")
    y_true = X

    # diva output
    with tf.name_scope("diva_outputs") as scope:
        decoder_op, accuracy_vec = divann.run(X)

    # prediction
    with tf.name_scope("prediction") as scope:
        current_class = tf.placeholder(tf.int32)
        y_pred = tf.cond(current_class > 0 , lambda: decoder_op[1], lambda: decoder_op[0])

    # accuracy
    with tf.name_scope("accuracy") as scope:
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

if __name__ == "__main__":
    main()

