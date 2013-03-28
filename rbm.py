'''

A simple Restricted Boltzmann Machine Implementation.

A probalistic generative model which will learn weightings
between a symmetrically connected layer of hidden and 
visible binary stochastic neurons so as to maximize 
the likelihood of generating output similar to input.

'''

import numpy as np
from time import time
import matplotlib.pyplot as plt

class rbm:

	def __init__(self,vSize,hSize,lr):
		self.h = np.zeros(hSize)
		self.v = np.zeros(vSize)
		self.lr = lr

	def energy(self,v,h,w):
		cbs = np.dstack(np.meshgrid(v,h)).reshape(-1, 2)
		return -(np.prod(cbs,axis=1)*w.flatten()).sum()

	''' Stochastic activation of hidden layer given input and weights '''
	def actH(self,v,h,w,stochastic=True):
		z = (v*w).sum(axis=1)
		prob = 1/(1+np.exp(-z))
		if stochastic:
			h = np.array((prob > np.random.random(h.size)),dtype=int)
		else:
			h = prob
		return h

	''' Stochasitic activation of visible layer given hidden and weights '''
	def actV(self,v,h,w,stochastic=True):
		z = (h*w.transpose()).sum(axis=1)
		prob = 1/(1+np.exp(-z))
		if stochastic:
			v = np.array((prob > np.random.random(v.size)),dtype=int)
		else:
			v = prob
		return v

	''' Quick single pass contrastive divergence to update weights '''
	def updateWeights(self,v,h,w,lr,stochastic=True):
		h = self.actH(v,h,w)
		start = np.prod(np.dstack(np.meshgrid(v,h)).reshape(-1, 2),axis=1)
		v = self.actV(v,h,w,stochastic)
		h = self.actH(v,h,w)
		end = np.prod(np.dstack(np.meshgrid(v,h)).reshape(-1, 2),axis=1)
		dw = lr*(start-end)
		w += dw.reshape((h.size,v.size))
		return w

	def randomSample(self,a):
		index = np.random.randint(0,a.shape[0])
		return a[index]

	def formatOutput(self,v):
		out = vect.inverse_transform(v)[0]
		outStr = ''
		for word in out:
			outStr += word + ' '
		return outStr

# t0 = time()
# trainNum = 1000
# testNum = 1000
# jobData = LoadData('Train_rev1.csv')
# sals = []
# titles = []
# for i in xrange(trainNum):
# 	job = jobData[i].data
# 	titles.append(job['title'])
# 	sals.append(np.log(float(job['salaryNormalized'])))
# vect = CountVectorizer(min_df=1,max_features=1000)
# vect = vect.fit(titles)
# XOrig = vect.transform(titles)
# print "done in %fs" % (time() - t0)
# print "n_samples: %d, n_features: %d" % XOrig.shape
# XFull = XOrig.toarray()
# print XFull.shape
# randomSample(XFull)

# lr = 0.005
# v = randomSample(XFull)
# h = np.zeros(1000)
# w = np.random.normal(loc=0,scale=0.01,size=(h.size,v.size))

# print 'learning rbm'

# for i in xrange(10000):
# 	v = randomSample(XFull)
# 	if i % 10 == 0: 
# 		inStr = formatOutput(v)
# 		# plt.imshow(actH(v,h,w,stochastic=False).reshape(25,-1),cmap='gray')
# 		# plt.plot(actH(v,h,w,stochastic=False).flatten())
# 		# plt.show()
# 		for j in xrange(200):
# 			h = actH(v,h,w)
# 			v = actV(v,h,w)
# 		# plt.hist(w.flatten(),bins=100)
# 		# plt.show()
# 		outStr = formatOutput(v)
# 		print 'input',inStr
# 		# plt.show()
# 		print 'iteration:',i,'output:',outStr
# 	# v = randomSample(XFull)
# 	w = updateWeights(v,h,w,lr)