import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

arr = np.loadtxt("test3_1.csv", delimiter=',', dtype=np.object, encoding='utf-8')
arr = MinMaxScaler().fit_transform(arr)
arr2 = np.loadtxt("test3_2.csv", delimiter=',', dtype=np.object, encoding='utf-8')
# arr2 = MinMaxScaler().fit_transform(arr2)

x_data = arr[0:, 1: -1]
# x_data = MinMaxScaler().fit_transform(x_data)
x_test = arr2[0:, 1:-1]

y_data = arr[0:, [-1]]

y_test = arr2[0:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
w_hist = tf.summary.histogram("weight", w)
b_hist = tf.summary.histogram("bias", b)

hypothesis = tf.matmul(x, w)
hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)


cost = tf.reduce_mean(tf.square(hypothesis - y))
cost_summ = tf.summary.scalar("cost", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

summary = tf.summary.merge_all()

prediction = tf.equal(hypothesis, y)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs")
    writer.add_graph(sess.graph)

    for epoch in range(30001) :
        c, s, _ = sess.run([cost, summary, optimizer], feed_dict={x: x_data, y: y_data})
        writer.add_summary(s, global_step=epoch)

        if (epoch % 1000 == 0):
            print("epoch : ", epoch, "cost : ", c)

    y_val, hy_val = sess.run([y, hypothesis], feed_dict={x: x_data, y: y_data})
    print("hy_val : ", hy_val, "\ny_val : ", y_val)
    print("Complete")