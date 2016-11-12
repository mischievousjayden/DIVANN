import tensorflow as tf

from data import Data 

# parameters
learning_rate = 0.01
training_epochs = 20
total_batch = 13
display_step = 1
examples_to_show = 10

# network parameters
n_hidden = 8
n_input = 3

# tf graph input
X = tf.placeholder("float", [None, n_input])

weights = {
    "encoder": tf.Variable(tf.random_normal([n_input, n_hidden])),
    "decoder": tf.Variable(tf.random_normal([n_hidden, n_input]))
}
biases = {
    "encoder": tf.Variable(tf.random_normal([n_hidden])),
    "decoder": tf.Variable(tf.random_normal([n_input]))
}

# building the encoder
def encoder(x):
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder"]), biases["encoder"]))
    return layer

def decoder(x):
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder"]), biases["decoder"]))
    return layer

# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# prediction
y_pred = decoder_op

# target
y_true = X

# define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# initializing the variables
init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)

for epoch in range(training_epochs):
    for i in range(total_batch):
        _, c = sess.run([optimizer, cost], feed_dict={X: Data.Stimuli[i]})
    if epoch % display_step == 0:
        print("Epoch:", ' %04d' % (epoch+1), "cost=", "{:.9f}".format(c))
print("Optimization Finished!")

