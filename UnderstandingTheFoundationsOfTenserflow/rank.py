import tensorflow as tf

sess = tf.Session()

zeroD = tf.constant(5)
sess.run(tf.rank(zeroD))

oneD = tf.constant(['how', 'are', 'you?'])
print(sess.run(tf.rank(oneD)))

twoD = tf.constant([[1, 3], [4, 6]])
print(sess.run(tf.rank(twoD)))

threeD = tf.constant([[[1,2], [6,7]], [[3,4], [9,0]]])
print(sess.run(tf.rank(threeD)))

session.close()