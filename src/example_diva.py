import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from data import Data
from diva.diva_origin import DIVANN

import pdb

class Logger:
    def __init__(self, filename):
        self.fp = open(filename, "a")

    def __del__(self):
        if self.fp != None:
            self.fp.close()

    def print(self, outstr):
        self.fp.write(outstr)
        # print(outstr, end="")

    def println(self, outstr):
        self.print(outstr+"\n")

def write_log(encoder_info, decoder_info, dirname, prefix):
    df_encoder_weight = pd.DataFrame(encoder_info["weight"])
    df_encoder_weight.to_csv("{}/{}_encoder_weight.csv".format(dirname, prefix), sep=',', index=False)

    df_encoder_bias = pd.DataFrame(encoder_info["bias"])
    df_encoder_bias.to_csv("{}/{}_encoder_bias.csv".format(dirname, prefix), sep=',', index=False)


    df_decoder_weight1 = pd.DataFrame(decoder_info["weights"][0])
    df_decoder_weight1.to_csv("{}/{}_decoder1_weight.csv".format(dirname, prefix), sep=',', index=False)

    df_decoder_bias1 = pd.DataFrame(decoder_info["biases"][0])
    df_decoder_bias1.to_csv("{}/{}_decoder1_bias.csv".format(dirname, prefix), sep=',', index=False)

    df_decoder_weight2 = pd.DataFrame(decoder_info["weights"][1])
    df_decoder_weight2.to_csv("{}/{}_decoder2_weight.csv".format(dirname, prefix), sep=',', index=False)

    df_decoder_bias2 = pd.DataFrame(decoder_info["biases"][1])
    df_decoder_bias2.to_csv("{}/{}_decoder2_bias.csv".format(dirname, prefix), sep=',', index=False)


def compare_df(df1, df2):
    same = (df1-df2).abs() < 0.0000001
    return False if same.sum().sum() == 0 else True

def compare_weights(pre_encoder_info, pre_decoder_info, encoder_info, decoder_info, message):

    if pre_encoder_info is None:
        return

    logger = Logger("same_weight.txt")

    df_pre_encoder_weight = pd.DataFrame(pre_encoder_info["weight"])
    df_encoder_weight = pd.DataFrame(encoder_info["weight"])
    if compare_df(df_pre_encoder_weight, df_encoder_weight):
        logger.println("{}_encoder_weight".format(message))

    df_pre_encoder_bias = pd.DataFrame(pre_encoder_info["bias"])
    df_encoder_bias = pd.DataFrame(encoder_info["bias"])
    if compare_df(df_pre_encoder_bias, df_encoder_bias):
        logger.println("{}_encoder_bias".format(message))

    df_pre_decoder_weight1 = pd.DataFrame(pre_decoder_info["weights"][0])
    df_decoder_weight1 = pd.DataFrame(decoder_info["weights"][0])
    if compare_df(df_pre_decoder_weight1, df_decoder_weight1):
        logger.println("{}_decoder1_weight".format(message))

    df_pre_decoder_bias1 = pd.DataFrame(pre_decoder_info["biases"][0])
    df_decoder_bias1 = pd.DataFrame(decoder_info["biases"][0])
    if compare_df(df_pre_decoder_bias1, df_decoder_bias1):
        logger.println("{}_decoder1_bias".format(message))

    df_pre_decoder_weight2 = pd.DataFrame(pre_decoder_info["weights"][1])
    df_decoder_weight2 = pd.DataFrame(decoder_info["weights"][1])
    if compare_df(df_pre_decoder_weight2, df_decoder_weight2):
        logger.println("{}_decoder2_weight".format(message))

    df_pre_decoder_bias2 = pd.DataFrame(pre_decoder_info["biases"][1])
    df_decoder_bias2 = pd.DataFrame(decoder_info["biases"][1])
    if compare_df(df_pre_decoder_bias2, df_decoder_bias2):
        logger.println("{}_decoder2_bias".format(message))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("beta", help="beta", type=float)
    parser.add_argument("learning_rate", help="learning rate", type=float)
    parser.add_argument("num_hidden", help="number of neurons in hidden layer", type=int)
    parser.add_argument("result_filename", help="result file name")

    args = parser.parse_args()
    result_filename = args.result_filename

    # parameters
    num_set_inputs = len(Data.Stimuli)
    training_epochs = 25
    display_step = 5

    # network parameters
    num_features = len(Data.Stimuli[0][0]) # 3
    num_hidden = args.num_hidden # 8
    num_classes = 2
    beta = args.beta # 0.8
    learning_rate = args.learning_rate # 0.01
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
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    avg_acc = list()
    cumulative_cost = list()

    # initializing the variables
    init = tf.initialize_all_variables()
    df_cost = pd.DataFrame()
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()

        encoder_info = None
        decoder_info = None
        
        pre_encoder_info = None
        pre_decoder_info = None

        for i in range(num_set_inputs):
            logfilename = "./logs/set{}".format(i)
            writer = tf.train.SummaryWriter(logfilename, sess.graph_def)
            sess.run(init)
            sum_acc = 0
            sum_cost = 0
            cost_list = list()
            for epoch in range(training_epochs):
                temp_sum_cost = 0
                for j, current_input in enumerate(Data.Stimuli[i]):
                    _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X:[current_input], current_class:Data.Assignments[j]})
                    # weight log
                    pre_encoder_info = encoder_info
                    pre_decoder_info = decoder_info

                    encoder_info = sess.run(divann._encoder)
                    decoder_info = sess.run(divann._decoder)

                    write_log(encoder_info, decoder_info, "./weight_log", "{}_{}_{}".format(i, epoch, j))

                    compare_weights(pre_encoder_info, pre_decoder_info, encoder_info, decoder_info, "{}_{}_{}".format(i, epoch, j))
 
                    temp_sum_cost = temp_sum_cost + (1-a)
                    if epoch == training_epochs-1:
                        sum_acc = sum_acc + a
                    if epoch % display_step == display_step-1:
                        sum_cost = sum_cost + (1-a)
                        summary = sess.run(merged, feed_dict={X:[current_input], current_class:Data.Assignments[j]})
                        writer.add_summary(summary, epoch)
                        print("input_set: {}".format(i), "Epoch: {}".format(epoch+1), "cost: {:.9f}".format(c), "accuracy={:.9f}".format(a), "label={}".format(Data.Assignments[j]))
                cost_list.append(temp_sum_cost / len(Data.Stimuli[i]))
            df_cost["stimuli{}".format(i)] = np.array(cost_list)
            avg_acc.append(sum_acc / len(Data.Stimuli[0]))
            cumulative_cost.append(sum_cost / len(Data.Stimuli[0]))
        print("Optimization Finished!")
        # plot
        columns = df_cost.columns.tolist()
        for i, col in enumerate(columns):
            data = df_cost[col].tolist()
            plt.plot(range(len(data)), data, label=col)
        plt.title("this is title")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        # plt.show()
        plt.savefig("plot_name.png")

        avg_acc = np.array(avg_acc)
        rank = np.absolute(avg_acc.argsort().argsort()-len(avg_acc))
        cumulative_cost = np.array(cumulative_cost)
        df = pd.DataFrame()
        df["cumulative_cost"] = cumulative_cost
        df["avg_acc"] = avg_acc
        df["rank"] = rank
        df.to_csv(result_filename, sep=',', index=False)
        print(df)

if __name__ == "__main__":
    main()

