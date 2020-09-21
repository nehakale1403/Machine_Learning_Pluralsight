import tensorflow as tf 
import matplotlib.image as mp_img
import matplotlib.pyplot as plot
import os

filename="./dandelion.jpg"

image = mp_img.imread(filename)

print('Image shape: ', image.shape)
print('Image array: ', image)

plot.imshow(image)
plot.show()

x = tf.Variable(image, name='x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	transpose = tf.transpose(x, perm = [1,0,2])# here we are swapping the !st and 2nd axis for getting transpose

	result = sess.run(transpose)

	print('Transposed image shape: ', result.shape)
	plot.imshow(result)
	plot.show()