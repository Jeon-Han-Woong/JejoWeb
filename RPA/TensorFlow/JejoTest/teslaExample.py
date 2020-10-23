import tensorflow as tf
import numpy as np

arr = np.loadtxt('output.txt', delimiter=',', dtype=np.object)
x_data = arr[1:, 0:-1]
y_data = arr[1:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([1, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
w_hist = tf.summary.histogram("weight", w)
b_hist = tf.summary.histogram("bias", b)

hypothesis = x * w + b
hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

cost = tf.reduce_mean(tf.square(hypothesis - y))
cost_summ = tf.summary.scalar("cost", cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs")
    writer.add_graph(sess.graph)
    for step in range(5001):
        cost_val, hy_val, w_val, s, _ = sess.run([cost, hypothesis, w, summary, optimizer], feed_dict={x: x_data, y: y_data})
        writer.add_summary(s, global_step=step)

        if step % 1000 == 0:
            print(step, "Cost : ", cost_val, "w : ", w_val)
