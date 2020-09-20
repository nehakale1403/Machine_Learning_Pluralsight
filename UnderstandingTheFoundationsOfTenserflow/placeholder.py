import tensorflow as tf 

x = tf.compat.v1.placeholder(tf.int32, shape=[3], name='x')
y = tf.compat.v1.placeholder(tf.int32, shape=[3], name='y')

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')

final_div = tf.div(sum_x, prod_y, name='final_div')
final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

sess = tf.Session()

print('sum(x): ', sess.run(sum_x, feed_dict = {x: [100,200,300]}))
print('prod(y): ', sess.run(prod_y, feed_dict = {y: [1,2,3]}))

writer = tf.compat.v1.summary.FileWriter('m3_example2', sess.graph)
writer.close()
sess.close()
