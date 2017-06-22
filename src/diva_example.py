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


def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def draw_plot(df, title, filename=None):
    columns = df.columns.tolist()
    for i, col in enumerate(columns):
        data = df[col].tolist()
        plt.plot(range(len(data)), data, label="{}".format(i+1))
    plt.title(title)
    plt.xlabel("Learning Epoch")
    plt.ylabel("Mean Classification Error Rate")
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()


def write_log(encoder_info, decoder_info, dirname, prefix):
    create_dir(dirname)

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

def compare_weights(label, pre_encoder_info, pre_decoder_info, encoder_info, decoder_info, message):

    if pre_encoder_info is None:
        return

    logger = Logger("{}/same_weight.txt".format(logdir))

    df_pre_encoder_weight = pd.DataFrame(pre_encoder_info["weight"])
    df_encoder_weight = pd.DataFrame(encoder_info["weight"])
    if compare_df(df_pre_encoder_weight, df_encoder_weight):
        logger.println("{}_encoder_weight".format(message))

    df_pre_encoder_bias = pd.DataFrame(pre_encoder_info["bias"])
    df_encoder_bias = pd.DataFrame(encoder_info["bias"])
    if compare_df(df_pre_encoder_bias, df_encoder_bias):
        logger.println("{}_encoder_bias".format(message))

    if label == 0:
        df_pre_decoder_weight1 = pd.DataFrame(pre_decoder_info["weights"][0])
        df_decoder_weight1 = pd.DataFrame(decoder_info["weights"][0])
        if compare_df(df_pre_decoder_weight1, df_decoder_weight1):
            logger.println("{}_decoder1_weight".format(message))

        df_pre_decoder_bias1 = pd.DataFrame(pre_decoder_info["biases"][0])
        df_decoder_bias1 = pd.DataFrame(decoder_info["biases"][0])
        if compare_df(df_pre_decoder_bias1, df_decoder_bias1):
            logger.println("{}_decoder1_bias".format(message))

    else:
        df_pre_decoder_weight2 = pd.DataFrame(pre_decoder_info["weights"][1])
        df_decoder_weight2 = pd.DataFrame(decoder_info["weights"][1])
        if compare_df(df_pre_decoder_weight2, df_decoder_weight2):
            logger.println("{}_decoder2_weight".format(message))

        df_pre_decoder_bias2 = pd.DataFrame(pre_decoder_info["biases"][1])
        df_decoder_bias2 = pd.DataFrame(decoder_info["biases"][1])
        if compare_df(df_pre_decoder_bias2, df_decoder_bias2):
            logger.println("{}_decoder2_bias".format(message))


def diva_sample(learning_rate, num_hidden, weight_range, beta, logdir):
    create_dir(logdir)
    conbination = "{}_{}_{}_{}".format(learning_rate, num_hidden, weight_range[1], beta)
    result_filename = "{}/{}".format(logdir, conbination)

    # parameters
    num_set_inputs = len(Data.Stimuli)
    training_epochs = 25
    display_step = 5

    # network parameters
    num_features = len(Data.Stimuli[0][0]) # 3
    num_classes = 2
    divann = DIVANN(num_features, num_hidden, weight_range, num_classes, beta)

    # tf graph input
    X = tf.placeholder("float", [None, num_features])
    y_true = X

    # diva output
    decoder_op, accuracy_vec = divann.run(X)

    # prediction
    current_class = tf.placeholder(tf.int32)
    y_pred = tf.cond(current_class > 0 , lambda: decoder_op[1], lambda: decoder_op[0])

    # accuracy
    accuracy = tf.cond(current_class > 0 , lambda: accuracy_vec[1], lambda: accuracy_vec[0])

    # define cost
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

    # Minimize
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    avg_acc = list()
    cumulative_cost = list()

    # initializing the variables
    init = tf.initialize_all_variables()
    df_cost = pd.DataFrame()
    with tf.Session() as sess:
        for i in range(num_set_inputs):
            sess.run(init)
            sum_acc = 0
            sum_cost = 0
            cost_list = list()

            encoder_info = None
            decoder_info = None
            
            pre_encoder_info = None
            pre_decoder_info = None

            for epoch in range(training_epochs):
                temp_sum_cost = 0
                for j, current_input in enumerate(Data.Stimuli[i]):
                    _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X:[current_input], current_class:Data.Assignments[j]})
                    # weight log
                    pre_encoder_info = encoder_info
                    pre_decoder_info = decoder_info

                    encoder_info = sess.run(divann._encoder)
                    decoder_info = sess.run(divann._decoder)

                    write_log(encoder_info, decoder_info, "{}/weight_log".format(logdir), "{}_{}_{}".format(i, epoch, j))

                    compare_weights(Data.Assignments[j], pre_encoder_info, pre_decoder_info, encoder_info, decoder_info, "{}_{}_{}".format(i, epoch, j))
 
                    temp_sum_cost = temp_sum_cost + (1-a)
                    if epoch == training_epochs-1:
                        sum_acc = sum_acc + a
                    if epoch % display_step == display_step-1:
                        sum_cost = sum_cost + (1-a)
                        print("input_set: {}".format(i), "Epoch: {}".format(epoch+1), "cost: {:.9f}".format(c), "accuracy={:.9f}".format(a), "label={}".format(Data.Assignments[j]))
                cost_list.append(temp_sum_cost / len(Data.Stimuli[i]))
            df_cost["stimuli{}".format(i)] = np.array(cost_list)
            avg_acc.append(sum_acc / len(Data.Stimuli[0]))
            cumulative_cost.append(sum_cost / len(Data.Stimuli[0]))

        print("Optimization Finished!")
        # plot
        draw_plot(df_cost, conbination, "{}.png".format(result_filename))

        avg_acc = np.array(avg_acc)
        rank = np.absolute(avg_acc.argsort().argsort()-len(avg_acc))
        rank = np.append(rank, 0)
        avg_acc = np.append(avg_acc, np.mean(avg_acc))
        cumulative_cost = np.append(cumulative_cost, np.mean(cumulative_cost))
        df_cost.to_csv("{}_cost.csv".format(result_filename), sep=',', index=False)

        df = pd.DataFrame()
        df["cumulative_cost"] = cumulative_cost
        df["avg_acc"] = avg_acc
        df["rank"] = rank
        # df.to_csv(result_filename, sep=',', index=False)
        df.to_csv("{}_result.csv".format(result_filename), sep=',', index=False)
        print(df)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="log directory")
    args = parser.parse_args()
    logroot = args.logdir

    learning_rate_set = [0.15]
    num_hidden_set = [4]
    weight_range_set = [2, 4]
    beta_set = [0.5]

    # learning_rate_set = [0.001, 0.05]
    # num_hidden_set = [4, 8]
    # weight_range_set = [2, 4]
    # beta_set = [0.5, 0.8]
    
    result = dict()
    for learning_rate in learning_rate_set:
        for num_hidden in num_hidden_set:
            for weight_range in weight_range_set:
                for beta in beta_set:
                    combination = "{}_{}_{}_{}".format(learning_rate, num_hidden, weight_range, beta)
                    print("[Start Conbination: {}]".format(combination))
                    logdir = "{}/{}".format(logroot, combination)
                    result[combination] = diva_sample(learning_rate, num_hidden, (0, weight_range), beta, logdir)
    result = pd.concat(result)
    result.to_csv("{}/overall_result.csv".format(logroot), sep=',')

