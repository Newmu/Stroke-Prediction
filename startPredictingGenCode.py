import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from time import time
from random import randint
from PIL import Image
import cPickle
import os

from skimage.io import imread
import skimage.exposure as Exposure
import skimage.filter as Filter
from skimage.morphology import skeletonize
from skimage.transform import resize
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_foerstner

from scipy.spatial.distance import cdist
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import convolve
from scipy import interpolate

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

def Neighbors(arr,x,y,n=3):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    return arr[:n,:n]

def getFilePath(n):
	base = 'images/'
	filepath = base+('0'*(4-len(str(n))))+str(n)+'.jpg'
	return filepath

def transform(n,rsizeP,padP):
	imgO = imread(getFilePath(n),as_grey=True)
	# thresh = Filter.threshold_otsu(imgO)
	thresh = 0.9
	img = imgO < thresh
	# img = skeletonize(img)
	y,x = np.nonzero(img)
	imgO = imgO[y.min():y.max(),x.min():x.max()]
	imgR = resize(imgO,(rsizeP,rsizeP))
	imgP = np.pad(imgR,(padP,padP),mode='constant',constant_values=1)
	return imgR,imgP,y.max()-y.min(),x.max()-x.min()

def pilResize(a,size):
	pilImg = Image.fromarray(a)
	pilImg = pilImg.resize(size, Image.ANTIALIAS)
	return np.array(pilImg)

def energy(v,h,w):
	cbs = np.dstack(np.meshgrid(v,h)).reshape(-1, 2)
	return -(np.prod(cbs,axis=1)*w.flatten()).sum()

''' Stochastic activation of hidden layer given input and weights '''
def actH(v,h,w,stochastic=True):
	z = (v*w).sum(axis=1)
	prob = 1/(1+np.exp(-z))
	if stochastic:
		h = np.array((prob > np.random.random(h.size)),dtype=int)
	else:
		h = prob
	return h

''' Stochasitic activation of visible layer given hidden and weights '''
def actV(v,h,w,stochastic=True):
	z = (h*w.transpose()).sum(axis=1)
	prob = 1/(1+np.exp(-z))
	if stochastic:
		v = np.array((prob > np.random.random(v.size)),dtype=int)
	else:
		v = prob
	return v

''' Quick single pass contrastive divergence to update weights '''
def updateWeights(v,h,w,lr,stochastic=True):
	h = actH(v,h,w)
	start = np.prod(np.dstack(np.meshgrid(v,h)).reshape(-1, 2),axis=1)
	v = actV(v,h,w,stochastic=False)
	h = actH(v,h,w,stochastic=False)
	end = np.prod(np.dstack(np.meshgrid(v,h)).reshape(-1, 2),axis=1)
	dw = lr*(start-end)
	w += dw.reshape((h.size,v.size))
	return w

def randomSample(a,b):
	index = np.random.randint(0,a.shape[0])
	img = a[index]
	path = b[index][:,0]
	return np.hstack((img,path.flatten()))

os.system("taskset -p 0xff %d" % os.getpid())

t0 = time()
print 'loading training csv'

reader = open('train.csv')
sigs = defaultdict(list)
for i,line in enumerate(reader):
	if i > 0:
		data = [field.strip() for field in line.split(',')]
		fields = ['row','sig','writer','occurence','time','x','y']
		data = dict(zip(fields,data))
		sigs[int(data['sig'])].append([float(data['x']),float(data['y']),float(data['time'])])

print 'loading test csv'
print 'loaded all data in',time()-t0

reader = open('test.csv')
tests = defaultdict(list)
for i,line in enumerate(reader):
	if i > 0:
		data = [field.strip() for field in line.split(',')]
		fields = ['row','sig','writer','occurence','time','x','y']
		data = dict(zip(fields,data))
		tests[int(data['sig'])].append([0,0,float(data['time'])])

# img = imread('rbmImgs/100Feats4.png',as_grey=True)
# img = Filter.median_filter(img,radius=7)
# plt.imshow(img,cmap='gray')
# plt.show()

# def formatDS(start,stop,rsizeP,padP):
# 	print start,stop
# 	X = []
# 	Y = []
# 	for i in xrange(start,stop):
# 		if i % 50 == 0: print i
# 		imgR,imgP,xS,yS = transform(i,rsizeP,padP)
# 		pX,pY = sigs[i][0][0],sigs[i][0][1]
# 		pX,pY = (padP+pX*rsizeP)/(padP*2+rsizeP),(padP+pY*rsizeP)/(padP*2+rsizeP)
# 		print pX,pY
# 		plt.imshow(imgP)
# 		plt.show()
# 		X.append(imgP.flatten())
# 		Y.append([pX,pY])
# 	return np.array(X),np.array(Y)

def formatDS(start,stop,r,useIntersects=True):
	aStarts = {}
	X = []
	Y = []
	coords = []
	sigNum = []
	strcArr = [[1,1,1],[1,1,1],[1,1,1]]
	for sig in xrange(start,stop):
		if sig % 50 == 0: print sig
		path = np.array(sigs[sig])
		imgO = imread(getFilePath(sig),as_grey=True)
		thresh = 0.9
		img = imgO < thresh
		imgX,imgY = np.nonzero(img)
		imgW = imgX.max()-imgX.min()
		imgH = imgY.max()-imgY.min()
		img = skeletonize(img)
		img = img[imgX.min():imgX.max(),imgY.min():imgY.max()]
		skltnSegs,nSkltnSegs = label(img,structure=strcArr)
		# print nSkltnSegs
		# plt.imshow(skltnSegs)
		# plt.show()
		imgO = imgO[imgX.min():imgX.max(),imgY.min():imgY.max()]
		sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
		summed = convolve(img,sumArr,mode='constant',cval=0)
		corners = ((summed == 2) & (img == 1))
		if useIntersects:
			intersects = ((summed >= 4) & (img == 1))
			labeled,nLabels = label(intersects,structure=strcArr)
			itrscts = []
			for l in xrange(1,nLabels+1):
				intersect = np.array((labeled == l),dtype=int)
				posX,posY = np.nonzero(intersect)
				xC,yC = np.array([[np.sum(posX)/posX.size],[np.sum(posY)/posY.size]])
				itrscts.append([xC[0],yC[0]])
			itrscts = np.array(itrscts)
		corners = np.transpose(np.array(np.nonzero(corners)))
		startx = path[0][0]*img.shape[1]
		starty = path[0][1]*img.shape[0]
		aStarts[sig] = [startx,starty]
		start = np.transpose(np.array([[startx],[starty]]))
		if useIntersects:
			try:
				corners = np.vstack((itrscts,corners))
			except:
				print 'something went wrong at',sig
		corners[:,[0,1]] = corners[:,[1,0]]
		dists = cdist(start,corners)
		if dists.min() < 15:
			rights = corners[(dists < 15).flatten()]
			if len(rights) > 0:
				for x,y in rights:
					# plt.imshow(img)
					# plt.scatter(x,y)
					# plt.show()
					correct = Neighbors(imgO,y-r/2,x-r/2,r).flatten()
					# plt.imshow(Neighbors(img,y-r/2,x-r/2,r))
					# plt.show()
					normedX = x/float(imgH)
					normedY = y/float(imgW)
					strokeLen = np.sum((skltnSegs == skltnSegs[y,x]))
					X.append(np.hstack((correct,[normedX,normedY,strokeLen])))
					# X.append(correct)
					Y.append(1)
					sigNum.append(sig)
					coords.append([x,y])
		else:
			x,y = corners[dists.argmin()][0],corners[dists.argmin()][1]
			correct = Neighbors(imgO,y-r/2,x-r/2,r).flatten()
			normedX = x/float(imgH)
			normedY = y/float(imgW)
			strokeLen = np.sum((skltnSegs == skltnSegs[y,x]))
			X.append(np.hstack((correct,[normedX,normedY,strokeLen])))
			# X.append(correct)
			Y.append(1)
			sigNum.append(sig)
			coords.append([x,y])
		# corners = np.delete(corners,dists.argmin(),axis=0)
		# plt.imshow(imgO,cmap='gray')
		# plt.scatter(x,y)
		# plt.show()
		wrongI = np.random.randint(0,corners.shape[0])
		# plt.imshow(img)
		for i,corner in enumerate(corners):
			# if i % 4 == 0:
			if dists[0][i] > 50:
				wrongX,wrongY = corner[0],corner[1]
				wrong = Neighbors(imgO,wrongY-r/2,wrongX-r/2,r).flatten()
				# X.append(wrong)
				# X.append([wrongX,wrongY])
				# plt.scatter(wrongX,wrongY)
				normedX = wrongX/float(imgH)
				normedY = wrongY/float(imgW)
				strokeLen = np.sum((skltnSegs == skltnSegs[wrongY,wrongX]))
				X.append(np.hstack((correct,[normedX,normedY,strokeLen])))
				Y.append(0)
				sigNum.append(sig)
				coords.append([x,y])
				# plt.subplot(1,2,1)
				# plt.imshow(Neighbors(imgO,y-r/2,x-r/2,r),cmap='gray')
				# plt.subplot(1,2,2)
				# plt.imshow(Neighbors(imgO,wrongY-r/2,wrongX-r/2,r),cmap='gray')
				# plt.show()
		# plt.show()
	return np.array(X),np.array(Y),np.array(sigNum),np.array(coords),aStarts

print 'formatting data'

d = 40
useIntersects = False
trainX,trainY,trainSigNum,trainCoords,trainStarts = formatDS(1,506,d,useIntersects)
validX,validY,validSigNum,validCoords,validStarts = formatDS(506,606,d,useIntersects)
# testX,testY = formatDS(601,606,50)
print trainX.shape,trainY.shape
print validX.shape,validY.shape
# print testX.shape,testY.shape
# train = (trainX,trainY)
# valid = (validX,validY)
# test = (testX,testY)
# ds = (train,valid,test)
# cPickle.dump(ds,open('ds.pkl','wb'))
print 'formated all data in',time()-t0

t1 = time()
# clf = svm.SVC(probability=True,verbose=True)
clf = RandomForestClassifier(n_estimators=2,verbose=2)
# clf = GradientBoostingClassifier(verbose=2)
# clf = linear_model.RidgeClassifier()
clf2 = KNeighborsClassifier()
print clf
clf.fit(trainX,trainY)
joblib.dump(clf, 'models/RF500TModel.skl', compress=3)
clf2.fit(trainX,trainY)
joblib.dump(clf2, 'models/KN500TModel.skl', compress=3)
print 'time to train',time()-t1

t2 = time()
predicted = clf.predict_proba(validX)
predicted2 = clf.predict_proba(validX)
predicted = (predicted*predicted2)
# print predicted
# print predicted[predicted != 0]
# print np.sum(predicted[predicted != 0])/float(np.sum(predicted[predicted == 0]))
print 'time to predict',time()-t2

correct = 0
wrong = 0
vStart = validSigNum.min()
vEnd = validSigNum.max()
cProbs = []
wProbs = []
for i in xrange(vStart,vEnd):
	poss = predicted[(validSigNum == i)]
	actual = validY[(validSigNum == i)]
	aCoords = validCoords[(validSigNum == i)]
	# maxLike = poss.max()
	# maxLikeIdx = poss.argmax()
	# print maxLike,maxLikeIdx,actual[maxLikeIdx]
	maxLike = poss[:,1].max()
	maxLikeIdx = poss[:,1].argmax()
	if actual[maxLikeIdx] == 1: 
		imgO = imread(getFilePath(i),as_grey=True)
		thresh = 0.9
		img = imgO < thresh
		imgX,imgY = np.nonzero(img)
		imgW = imgX.max()-imgX.min()
		imgH = imgY.max()-imgY.min()
		imgO = imgO[imgX.min():imgX.max(),imgY.min():imgY.max()]
		plt.imshow(imgO,cmap='gray')
		x,y = aCoords[maxLikeIdx]
		plt.scatter(x,y,c='green')
		plt.scatter(validStarts[i][0],validStarts[i][1],c='red')
		print 'correct'
		plt.show()
		correct += 1
		cProbs.append(maxLike)
	else:
		imgO = imread(getFilePath(i),as_grey=True)
		thresh = 0.9
		img = imgO < thresh
		imgX,imgY = np.nonzero(img)
		imgW = imgX.max()-imgX.min()
		imgH = imgY.max()-imgY.min()
		imgO = imgO[imgX.min():imgX.max(),imgY.min():imgY.max()]
		plt.imshow(imgO,cmap='gray')
		x,y = aCoords[maxLikeIdx]
		plt.scatter(x,y,c='green')
		plt.scatter(validStarts[i][0],validStarts[i][1],c='red')
		print 'wrong'
		plt.show()
		wrong += 1
		wProbs.append(maxLike)
		# print 'wrong'
print 'prob correct model:',correct/float(vEnd-vStart)
starts = np.sum(validY == 1)
wrongs = np.sum(validY == 0)
print 'prob correct random:',starts/float((starts+wrongs))
print 'diameter used',d,'used intersects',useIntersects
print 'total run time',time()-t0
plt.subplot(1,2,1)
plt.hist(cProbs,bins=10)
plt.subplot(1,2,2)
plt.hist(wProbs,bins=10)
plt.show()

# imgs = []
# paths = []
# # # for i in xrange(1,len(sigs)):
# for i in xrange(len(sigs)):
# 	i = randint(0,len(sigs))
# 	print i
# 	path = np.array(sigs[i])
# 	imgO = imread(getFilePath(i),as_grey=True)
# 	# thresh = Filter.threshold_otsu(imgO)
# 	thresh = 0.9
# 	img = imgO < thresh
# 	y,x = np.nonzero(img)
# 	img = skeletonize(img)
# 	# y,x = np.nonzero(img)
# 	img = img[y.min():y.max(),x.min():x.max()]
# 	# imgO = imgO[y.min():y.max(),x.min():x.max()]
# 	sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
# 	summed = convolve(img,sumArr,mode='constant',cval=0)
# 	# corners = (((summed == 2) | (summed > 4)) & (img == 1))
# 	corners = ((summed == 2) & (img == 1))
# 	intersects = ((summed >= 4) & (img == 1))
# 	corners = np.transpose(np.array(np.nonzero(corners)))
# 	print corners.shape
# 	plt.imshow(imgO,cmap='gray')
# 	# sX = path[0][0]*img.shape[1]+x.min()
# 	# sY = path[0][1]*img.shape[0]+y.min()
# 	sX = path[:,0]*img.shape[1]+x.min()
# 	sY = path[:,1]*img.shape[0]+y.min()
# 	plt.scatter(sX,sY,c='red',s=50)
# 	plt.scatter(corners[:,1]+x.min(),corners[:,0]+y.min(),c='green',s=50)
# 	plt.show()
# 	# corners[:,1] += x.min()
# 	# corners[:,0] += y.min()
# 	# img01 = pilResize(imgO,(100,100))
# 	# img02 = resize(imgO,(100,100))
# 	# img01 = np.array((img01 > thresh),dtype=int)
# 	# img02 = np.array((img02 > thresh),dtype=int)
# 	path[:,0] = path[:,0]*imgO.shape[1]
# 	path[:,1] = path[:,1]*imgO.shape[0]
# 	path[:,2] = path[:,2]/path[:,2].max()
# 	t = np.copy(path[:,2])
# 	x = np.copy(path[:,0])
# 	y = np.copy(path[:,1])
# 	mask = ((np.diff(x) == 0) & (np.diff(y) == 0))
# 	t[mask] += np.random.random(len(t[mask]))/10000.0
# 	x[mask] += np.random.random(len(t[mask]))/100.0
# 	y[mask] += np.random.random(len(t[mask]))/100.0
# 	tck,u = interpolate.splprep([x,y],k=1,s=0.1)
# 	tnew = np.linspace(0,1,300)
# 	outx,outy = interpolate.splev(tnew,tck)
# 	# tck,u = interpolate.bisplrep(x,y,t)
# 	# unew = np.linspace(0,1,300)
# 	# outx,outy = interpolate.bisplev(unew,tck)
# 	# print outx
# 	# print outy
# 	# plt.imshow(imgO,cmap='gray')
# 	# plt.plot(path[:,0],path[:,1],'r')
# 	# plt.plot(outx,outy,'b')
# 	paths.append([outx,outy,tnew])
# 	# plt.scatter(corners[:,1],corners[:,0],s=50,c='red')
# 	# imgs.append(img02.flatten())
# 	# plt.show()
# paths = np.array(paths)
# imgs = np.array(imgs)

# imgs = cPickle.load(open('imgs100pxFull.p','rb'))
# imgs = imgs[:604]
# print 'imgs loaded'
# paths = cPickle.load(open('pathsNormedLinInterp.p','rb'))
# print 'paths loaded'

# pathsNew = []
# for path in paths:
# 	# plt.plot(path[0,:],path[1,:])
# 	# plt.show()
# 	path[0,:] = path[0,:]/path[0,:].max()
# 	path[1,:] = path[1,:]/path[1,:].max()
# 	# plt.imshow(imgs[0].reshape(100,100))
# 	# plt.plot(path[0,:],path[1,:])
# 	# plt.show()
# 	pathsNew.append(path)
# cPickle.dump(pathsNew,open('pathsNormedLinInterp.p','wb'))
# print imgs.shape
# cPickle.dump(imgs,open('imgs100pxFull.p','wb'))
# cPickle.dump(paths,open('pathsLinInterp.p','wb'))

# lr = 0.01
# v = randomSample(imgs,paths)
# print v
# print v.shape
# h = np.zeros(2000)
# w = np.random.normal(loc=0,scale=0.01,size=(h.size,v.size))

# for i in xrange(10000):
# 	print i
# 	v = randomSample(imgs,paths)
# 	if i > 1000: 
# 		realPath = v[10000:].reshape(3,)
# 		plt.imshow(v[:10000].reshape(100,100),cmap='gray')
# 		# plt.imshow(actH(v,h,w,stochastic=False).reshape(25,-1),cmap='gray')
# 		# plt.plot(actH(v,h,w,stochastic=False).flatten())
# 		# plt.show()
# 		h = actH(v,h,w,stochastic=False)
# 		v = actV(v,h,w,stochastic=False)
# 		path = v[10000:].reshape(3,)
# 		print path
# 		# plt.hist(w.flatten(),bins=100)
# 		# plt.show()
# 		# plt.imshow(v[:10000].reshape(100,100),cmap='gray')
# 		plt.scatter(path[0]*100,path[1]*100,s=50,c='red')
# 		plt.scatter(realPath[0]*100,realPath[1]*100,s=50,c='green')
# 		plt.show()
# 	v = randomSample(imgs,paths)
# 	w = updateWeights(v,h,w,lr,stochastic=True)

# starts = []
# for i in xrange(1,len(sigs)):
# 	starts.append(sigs[i][0])
# starts = np.array(starts)
# plt.scatter(starts[:,0],1-starts[:,1])
# plt.show()
# T = np.zeros((100,100))
# num = 1010
# imgO = imread('images/0010.jpg',as_grey=True)
# thresh = Filter.threshold_otsu(imgO)
# thresh = 0.9
# img = imgO < thresh
# img = skeletonize(img)
# plt.subplot(1,2,1)
# plt.imshow(imgO,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img,cmap='gray')
# plt.show()

def getCorners(imgO):
	thresh = 0.9
	img = imgO < thresh
	y,x = np.nonzero(img)
	img = skeletonize(img)
	img = img[y.min():y.max(),x.min():x.max()]
	imgO = imgO[y.min():y.max(),x.min():x.max()]
	sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
	summed = convolve(img,sumArr,mode='constant',cval=0)
	corners = ((summed == 2) & (img == 1))
	corners = np.transpose(np.array(np.nonzero(corners)))
	corners[:,[0,1]] = corners[:,[1,0]]
	return corners

# r = 40
# ds = SupervisedDataSet(r*r, 1)
# X = []
# Y = []
# # for i in xrange(1,len(sigs)):
# # for i in xrange(1,len(sigs)-100):
# trainNum = 500
# testNum = 100
# for i in xrange(1,trainNum):
# 	if i % 10 == 0: print i
# 	path = np.array(sigs[i])
# 	imgO = imread(getFilePath(i),as_grey=True)
# 	# thresh = Filter.threshold_otsu(imgO)
# 	thresh = 0.9
# 	img = imgO < thresh
# 	y,x = np.nonzero(img)
# 	img = skeletonize(img)
# 	# y,x = np.nonzero(img)
# 	img = img[y.min():y.max(),x.min():x.max()]
# 	imgO = imgO[y.min():y.max(),x.min():x.max()]
# 	sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
# 	summed = convolve(img,sumArr,mode='constant',cval=0)
# 	# corners = (((summed == 2) | (summed > 4)) & (img == 1))
# 	corners = ((summed == 2) & (img == 1))
# 	intersects = ((summed >= 4) & (img == 1))
# 	corners = np.transpose(np.array(np.nonzero(corners)))
# 	startx = path[0][0]*img.shape[1]
# 	starty = path[0][1]*img.shape[0]
# 	start = np.transpose(np.array([[startx],[starty]]))
# 	corners[:,[0,1]] = corners[:,[1,0]]
# 	dists = cdist(start,corners)
# 	if dists.min() < 15:
# 		x,y = corners[dists.argmin()][0],corners[dists.argmin()][1]
# 		# corners = np.delete(corners,dists.argmin(),axis=0)
# 	else:
# 		x,y = int(startx),int(starty)
# 	# plt.imshow(imgO,cmap='gray')
# 	# plt.scatter(x,y)
# 	# plt.show()
# 	correct = Neighbors(imgO,y-r/2,x-r/2,r).flatten()
# 	wrongI = np.random.randint(0,corners.shape[0])
# 	for i,corner in enumerate(corners):
# 		# if i % 4 == 0:
# 		if dists[0][i] > 50:
# 			wrongX,wrongY = corner[0],corner[1]
# 			wrong = Neighbors(imgO,wrongY-r/2,wrongX-r/2,r).flatten()
# 			# X.append(wrong)
# 			# X.append([wrongX,wrongY])
# 			X.append(np.hstack((wrong,[wrongX,wrongY])))
# 			Y.append([0.0])
# 			# plt.subplot(1,2,1)
# 			# plt.imshow(Neighbors(imgO,y-r/2,x-r/2,r),cmap='gray')
# 			# plt.subplot(1,2,2)
# 			# plt.imshow(Neighbors(imgO,wrongY-r/2,wrongX-r/2,r),cmap='gray')
# 			# plt.show()
# 			ds.addSample(tuple(wrong),(-1))
# 	ds.addSample(tuple(correct),(1))
# 	# X.append(correct)
# 	# X.append([x,y])
# 	print x,y
# 	X.append(np.hstack((correct,[x,y])))
# 	Y.append([1.0])

# # net = buildNetwork(ds.indim, 100, ds.outdim, bias=True, hiddenclass=TanhLayer)
# # t = BackpropTrainer(net, ds, verbose = True)
# # # t.trainUntilConvergence()
# # t.trainOnDataset(ds, 10)

# X = np.array(X)
# Y = np.array(Y)
# print 'classifying SVM'
# print X.shape
# print Y.shape
# clf = svm.SVC(probability=True)
# clf.fit(X,Y)

# NTWrongs = 0.0
# NTCorrects = 0.0
# corrects = 0.0
# for i in xrange(len(sigs)-testNum+1,len(sigs)+1):
# 	print i
# 	path = np.array(sigs[i])
# 	imgO = imread(getFilePath(i),as_grey=True)
# 	# thresh = Filter.threshold_otsu(imgO)
# 	thresh = 0.9
# 	img = imgO < thresh
# 	y,x = np.nonzero(img)
# 	img = skeletonize(img)
# 	# y,x = np.nonzero(img)
# 	img = img[y.min():y.max(),x.min():x.max()]
# 	imgO = imgO[y.min():y.max(),x.min():x.max()]
# 	sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
# 	summed = convolve(img,sumArr,mode='constant',cval=0)
# 	# corners = (((summed == 2) | (summed > 4)) & (img == 1))
# 	corners = ((summed == 2) & (img == 1))
# 	intersects = ((summed >= 4) & (img == 1))
# 	corners = np.transpose(np.array(np.nonzero(corners)))
# 	startx = path[0][0]*img.shape[1]
# 	starty = path[0][1]*img.shape[0]
# 	start = np.transpose(np.array([[startx],[starty]]))
# 	corners[:,[0,1]] = corners[:,[1,0]]
# 	dists = cdist(start,corners)
# 	x,y = corners[dists.argmin()][0],corners[dists.argmin()][1]
# 	# if dists.min() < 15:
# 	# 	x,y = corners[dists.argmin()][0],corners[dists.argmin()][1]
# 	# 	# corners = np.delete(corners,dists.argmin(),axis=0)
# 	# else:
# 	# 	x,y = int(startx),int(starty)
# 	# plt.imshow(imgO,cmap='gray')
# 	# plt.scatter(x,y)
# 	# plt.show()
# 	correct = Neighbors(imgO,y-r/2,x-r/2,r).flatten()
# 	NTCorrects += 1
# 	# correctAs = net.activate(correct)
# 	# correctAs = clf.predict_proba(correct)
# 	# correctAs = clf.predict_proba([x,y])
# 	correctAs = clf.predict_proba(np.hstack((correct,[x,y])))
# 	wrongI = np.random.randint(0,corners.shape[0])
# 	wrongAs = []
# 	for i,corner in enumerate(corners):
# 		# if i % 4 == 0:
# 		if dists[0][i] > 20:
# 			wrongX,wrongY = corner[0],corner[1]
# 			wrong = Neighbors(imgO,wrongY-r/2,wrongX-r/2,r).flatten()
# 			# wrongAs.append(net.activate(wrong))
# 			# wrongAs.append(clf.predict_proba(wrong))
# 			# wrongAs.append(clf.predict_proba([wrongX,wrongY]))
# 			wrongAs.append(clf.predict_proba(np.hstack((wrong,[wrongX,wrongY]))))
# 			NTWrongs += 1
# 		# plt.subplot(1,2,1)
# 		# plt.imshow(Neighbors(imgO,y-r/2,x-r/2,r),cmap='gray')
# 		# plt.subplot(1,2,2)
# 		# plt.imshow(Neighbors(imgO,wrongY-r/2,wrongX-r/2,r),cmap='gray')
# 		# plt.show()
# 	wrongAs = np.array(wrongAs)
# 	# print wrongAs.shape
# 	# print wrongAs
# 	# print correctAs
# 	# print wrongAs.max(),wrongAs.argmax()
# 	# print 'correct prob',correctAs[0][1]
# 	# print 'wrong prob',wrongAs[:,:,1].max()
# 	if correctAs[0][1] >= wrongAs[:,:,1].max():
# 		print 'correct'
# 		print correctAs[0][1]
# 		print wrongAs[:,:,1].max()
# 		corrects += 1
# 	else:
# 		print 'wrong'
# 		print correctAs[0][1]
# 		print wrongAs[:,:,1].max()
# print corrects
# print NTCorrects,NTWrongs
# print 'random prob correct',(NTCorrects/(NTCorrects+NTWrongs))
# print 'actual prob correct',(corrects/NTCorrects)
# # print net
# # print t

# X = np.array(X)
# Y = np.array(Y)
# print X.shape
# print Y.shape
# lr = linear_model.LogisticRegression()
# lr.fit(X,Y)
# errs = []
# corrects = []
# wrongs = []
# for i in xrange(500,600):
# 	print i
# 	path = np.array(sigs[i])
# 	imgO = imread(getFilePath(i),as_grey=True)
# 	# thresh = Filter.threshold_otsu(imgO)
# 	thresh = 0.9
# 	img = imgO < thresh
# 	y,x = np.nonzero(img)
# 	img = skeletonize(img)
# 	# y,x = np.nonzero(img)
# 	img = img[y.min():y.max(),x.min():x.max()]
# 	# imgO = imgO[y.min():y.max(),x.min():x.max()]
# 	sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
# 	summed = convolve(img,sumArr,mode='constant',cval=0)
# 	corners = (((summed == 2) | (summed > 4)) & (img == 1))
# 	corners = np.transpose(np.array(np.nonzero(corners)))
# 	corners[:,1] += x.min()
# 	corners[:,0] += y.min()
# 	# plt.scatter(corners[:,1],corners[:,0],s=20)
# 	startx = path[0][0]*img.shape[1]+x.min()
# 	starty = path[0][1]*img.shape[0]+y.min()
# 	start = np.transpose(np.array([[startx],[starty]]))
# 	corners[:,[0,1]] = corners[:,[1,0]]
# 	dists = cdist(start,corners)
# 	if dists.min() < 15:
# 		x,y = corners[dists.argmin()][0],corners[dists.argmin()][1]
# 		corners = np.delete(corners,dists.argmin(),axis=0)
# 	else:
# 		x,y = int(startx),int(starty)
# 	r = 100
# 	correct = Neighbors(imgO,y-r/2,x-r/2,r)
# 	wrongI = np.random.randint(0,corners.shape[0])
# 	wrongX,wrongY = corners[wrongI][0],corners[wrongI][1]
# 	wrong = Neighbors(imgO,wrongY-r/2,wrongX-r

# errX = 0
# errY = 0
# for sig in sigs.values():
# 	sig = np.array(sig)
# 	predicted = np.ones_like(sig[:,:2])/2.0
# 	errX += np.sqrt(np.sum(np.square(predicted[:,0]-sig[:,0]))/len(sig))
# 	errY += np.sqrt(np.sum(np.square(predicted[:,1]-sig[:,1]))/len(sig))
# errX = errX/len(sigs)
# errY = errY/len(sigs)
# print (errX+errY)/2
# print errX,errY

# print len(tests)

# xP = {}
# yP = {}
# for i in xrange(len(sigs)+1,1082):
# # for i in xrange(1,len(sigs)+1):
# 	if i % 50 == 0: print i
# 	imgO = imread(getFilePath(i),as_grey=True)
# 	thresh = 0.5
# 	img = imgO < thresh
# 	# print imgO
# 	x,y = np.nonzero(img)
# 	xR = x.max()-x.min()
# 	yR = y.max()-y.min()
# 	xA = np.mean(x)
# 	yA = np.mean(y)
# 	# xIdxs = [n for n in xrange(0,img.shape[0])]
# 	# yIdxs = [n for n in xrange(0,img.shape[1])]
# 	# idxGrid = np.array(np.meshgrid(xIdxs,yIdxs)).T.reshape(-1,2)
# 	# # print idxGrid
# 	# intensities = (1-(imgO.flatten()/1.01))
# 	# xA = np.average(idxGrid[:,0],weights=intensities)
# 	# yA = np.average(idxGrid[:,1],weights=intensities)
# 	xGuess = (xA-x.min())/xR
# 	yGuess = (yA-y.min())/yR
# 	xP[i] = xGuess
# 	yP[i] = yGuess

# reader = open('testTest.csv')
# outWriter = csv.writer(open('testTestSubmission.csv','wb'))
# tests = defaultdict(list)
# for i,line in enumerate(reader):
# 	if i == 0:
# 		outWriter.writerow(line.split(','))
# 	if i > 0:
# 		data = [field.strip() for field in line.split(',')]
# 		data[5] = yP[int(data[1])]
# 		data[6] = xP[int(data[1])]
# 		outWriter.writerow(data)

# errX = 0
# errY = 0
# for i,sig in enumerate(sigs.values()):
# 	if i % 50 == 0: print i
# 	sig = np.array(sig)
# 	predicted = np.ones_like(sig[:,:2])
# 	predicted[:,0] = predicted[:,0]*yP[i+1]
# 	predicted[:,1] = predicted[:,1]*xP[i+1]
# 	errX += np.sqrt(np.sum(np.square(predicted[:,0]-sig[:,0]))/len(sig))
# 	errY += np.sqrt(np.sum(np.square(predicted[:,1]-sig[:,1]))/len(sig))
# errX = errX/len(sigs)
# errY = errY/len(sigs)
# print (errX+errY)/2
# print errX,errY
