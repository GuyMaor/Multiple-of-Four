import numpy as np
import tensorflow as tf
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ACTUAL HOMEWORK

def getVector(inLine):
	inLine = inLine.replace("\n","")
	strVal = inLine.split(',')
	inputs = [float(x) for x in strVal]
	labels = inputs[0]
	return inputs[1:],labels


ftes=open("data_test.txt", "r")
xtes = []
ytes = []
for line in ftes:
	xm,ym = getVector(line)
	ym = [ym]
	xtes.append(xm)
	ytes.append(ym)
xtes = np.asarray(xtes)
ytes = np.asarray(ytes)
dataSize = xtes[0].size
numPoints = np.size(xtes,0)
N_tes = np.size(xtes,0)

tf.reset_default_graph()

D_h = 5
with tf.device('/gpu:0'):
	x = tf.placeholder(tf.float32,[None,10],name="x")
	hidden = tf.layers.dense(inputs=x, units=D_h, activation=tf.nn.sigmoid,use_bias=True,name="hidden")
	h = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.sigmoid,use_bias=True,name="h")
pred = 0
config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config = config) as session:
	saver = tf.train.Saver()
	saver.restore(session, "/tmp/model.ckpt")

	w1 = tf.get_default_graph().get_tensor_by_name(os.path.split(hidden.name)[0] + '/kernel:0')
	w2 = tf.get_default_graph().get_tensor_by_name(os.path.split(h.name)[0] + '/kernel:0')
	b1 = tf.get_default_graph().get_tensor_by_name(os.path.split(hidden.name)[0] + '/bias:0')
	b2 = tf.get_default_graph().get_tensor_by_name(os.path.split(h.name)[0] + '/bias:0')

	w1 = session.run(w1)
	w2 = session.run(w2)
	b1 = session.run(b1)
	b2 = session.run(b2)	

	print("B1:",b1)
	print("W1:",w1)
	print("B2:",b2)
	print("W2:",w2)
	pred = session.run(h,feed_dict={x:xtes})
	pred = 1*(pred > 0.5)

labels = np.transpose(ytes)[0]
pred = np.transpose(pred)[0]

print("Labels:",labels)
print("Predictions:",pred)
num_wrong = np.sum(1*(labels != pred))
error = num_wrong/N_tes*100
print("Error %:",error)

