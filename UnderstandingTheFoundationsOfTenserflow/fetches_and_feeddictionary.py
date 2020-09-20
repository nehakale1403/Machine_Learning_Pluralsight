import tensorflow as tf 

# y = Wx + b
W = tf.constant([10,100], name = 'const_W')
x = tf.compat.v1.placeholder(tf.int32, name='x')
b = tf.compat.v1.placeholder(tf.int32, name='b')

Wx = tf.multiply(W, x, name='Wx')

y = tf.add(Wx, b, name='y')

y_ = tf.subtract(x, b, name='y_')

with tf.Session() as sess:
	print('Wx: ', sess.run(Wx, feed_dict={x: [3,33]})) #here Wx is the fetch
	print('Wx + b: ', sess.run(y, feed_dict={x: [5, 50], b: [7,9]}))
	print('Two results, Wx+b, x-b: ', sess.run(fetches=[y, y_], feed_dict={x:[5, 50], b:[7, 9]}))

