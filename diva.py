import tensorflow as tf

from data import Data 

# parameters
training_epochs = 20
total_batch = 13
display_step = 10

# network parameters
num_features = 3
num_hidden = 8

n_input = 3
n_hidden = 8

# tf graph input
X = tf.placeholder("float", [None, num_features])

weights = {
    "encoder": tf.Variable(tf.random_normal([num_features, num_hidden])),
    "decoder1": tf.Variable(tf.random_normal([num_hidden, num_features])),
    "decoder2": tf.Variable(tf.random_normal([num_hidden, num_features]))
}

biases = {
    "encoder": tf.Variable(tf.random_normal([num_hidden])),
    "decoder1": tf.Variable(tf.random_normal([num_features])),
    "decoder2": tf.Variable(tf.random_normal([num_features]))
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

def getPrediction(outputs, current_class):
    return outputs[current_class]

# construct model
encoder_op = encode(X)
decoder_op = decodeToMultiDim(encoder_op)

# prediction
current_class = tf.placeholder(tf.int32)
y_pred = tf.cond(current_class > 0 , lambda: decoder_op[1], lambda: decoder_op[0])

# target
y_true = X

# define loss and optimizer, minimize the squared error
learning_rate = 0.01
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# initializing the variables
init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)

for i in range(total_batch):
    for epoch in range(training_epochs):
        for j, current_input in enumerate(Data.Stimuli[i]):
            _, c = sess.run([optimizer, cost], feed_dict={X:[current_input], current_class:Data.Assignments[j]})
            if epoch % display_step == 0:
                # print(sess.run([X, encoder_op], feed_dict={X:[current_input]}))
                print("input:", i, "Epoch:", ' %04d' % (epoch+1), "cost=", "{:.9f}".format(c), "label=", Data.Assignments[j])
print("Optimization Finished!")

