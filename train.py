import numpy as np
import tensorflow as tf
import math


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def getVector(inLine):
	inLine = inLine.replace("\n","")
	strVal = inLine.split(',')
	inputs = [float(x) for x in strVal]
	labels = inputs[0]
	return inputs[1:],labels



ftrn=open("data_train.txt", "r")
xtrn = []
ytrn = []
for line in ftrn:
	xm,ym = getVector(line)

	ym = [ym]
	xtrn.append(xm)
	ytrn.append(ym)
xtrn = np.asarray(xtrn)
ytrn = np.asarray(ytrn)
dataSize = xtrn[0].size
numPoints = np.size(xtrn,0)

ftes=open("data_val.txt", "r")
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



def train_NN(D_h,xtrn,ytrn,xtes,ytes):
	N = np.size(xtrn,0)
	N_tes = np.size(xtes,0)
	x = tf.placeholder(tf.float32,[None,10],name="x")
	y = tf.placeholder(tf.float32,[None,1],name="y")
	hidden = tf.layers.dense(inputs=x, units=D_h, activation=tf.nn.sigmoid,use_bias=True,name="hidden")
	h = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.sigmoid,use_bias=True,name="h")
	err = tf.square(h-y)
	train_op = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.05).minimize(err)
	init = tf.global_variables_initializer()


	num_wrong = 0
	with tf.Session() as session:
		session.run(init)
		NUM_ITR = 1000
		NUM_BCH = 100
		delt = int(N/NUM_BCH)
		errv = 100
		while(errv>2):
			for bch in range(NUM_BCH):
				session.run(train_op,feed_dict={x:xtrn[delt*bch:(delt+1)*bch],y:ytrn[delt*bch:(delt+1)*bch]})	
			pred = session.run(h,feed_dict={x:xtes})
			pred = 1*(pred > 0.5)
			ylab = np.transpose(ytes)[0]
			predT = np.transpose(pred)[0]
			comp = np.asarray([ylab,predT])
			comp = np.transpose(comp)
			diff = 1*(ytes != pred)
			num_wrong = np.sum(diff)
			errv = num_wrong/N_tes*100
			print("Error %:",errv)
		saver = tf.train.Saver()
		save_path = saver.save(session, "/tmp/model.ckpt")
		print("Model saved in path: %s" % save_path)



with tf.device('/cpu:0'):
	train_NN(5,xtrn,ytrn,xtes,ytes)
print("Done!")


