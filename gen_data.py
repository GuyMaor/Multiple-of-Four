import random
import numpy as np

NUM_DATA = 10000



def rand_data(N):
	if(N<5):
		w = np.zeros(11,dtype=int)
		while(np.sum(w) != N):
			ind = np.random.randint(1,11,1)[0]
			w[ind] = 1
		return w		
	else:
		w = np.ones(11,dtype=int)
		w[0] = 0
		while(np.sum(w) != N):
			ind = np.random.randint(1,11,1)[0]
			w[ind] = 0
		return w


ftrn = open("data_train.txt", "w")
for i in range(NUM_DATA):
	w = rand_data(i%10)
	v = np.sum(w)
	if(v == 0 or v == 4 or v == 8):
		w[0] = 1
	for j in range(10):
		ftrn.write("{},".format(w[j]))
	ftrn.write("{}\n".format(w[10]))


for i in range(int(NUM_DATA*0.4)):
	w = rand_data((1+i%2)*4)
	w[0] = 1
	for j in range(10):
		ftrn.write("{},".format(w[j]))
	ftrn.write("{}\n".format(w[10]))

ftrn.close()



fval = open("data_val.txt","w")
for i in range(NUM_DATA):
	w = 1*(np.random.random_sample(11) > 0.5)
	w[0] = 0
	npsum = np.sum(w)
	if(npsum == 0 or npsum == 4 or npsum == 8):
		w[0] = 1
	for j in range(10):
		fval.write("{},".format(w[j]))
	fval.write("{}\n".format(w[10]))

fval.close

ftes = open("data_test.txt","w")
for i in range(NUM_DATA):
	w = 1*(np.random.random_sample(11) > 0.5)
	w[0] = 0
	npsum = np.sum(w)
	if(npsum == 0 or npsum == 4 or npsum == 8):
		w[0] = 1
	for j in range(10):
		ftes.write("{},".format(w[j]))
	ftes.write("{}\n".format(w[10]))

ftes.close
