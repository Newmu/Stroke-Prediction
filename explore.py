import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import skimage.exposure as Exposure
import skimage.filter as Filter
from skimage.morphology import skeletonize
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn import linear_model
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_foerstner
from scipy.ndimage.filters import convolve
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def Neighbors(arr,x,y,n=3):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    return arr[:n,:n]

def getFilePath(n):
	base = 'images/'
	filepath = base+('0'*(4-len(str(n))))+str(n)+'.jpg'
	return filepath

def transform(n):
	imgO = imread(getFilePath(i),as_grey=True)
	# thresh = Filter.threshold_otsu(imgO)
	thresh = 0.9
	img = imgO < thresh
	# img = skeletonize(img)
	y,x = np.nonzero(img)
	imgO = imgO[y.min():y.max(),x.min():x.max()]
	return resize(imgO,(100,100))

reader = open('train.csv')
sigs = defaultdict(list)
for i,line in enumerate(reader):
	if i > 0:
		data = [field.strip() for field in line.split(',')]
		fields = ['row','sig','writer','occurence','time','x','y']
		data = dict(zip(fields,data))
		sigs[int(data['sig'])].append([float(data['x']),float(data['y']),float(data['time'])])
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
X = []
Y = []
# for i in xrange(1,len(sigs)):
ds = SupervisedDataSet(10000, 1)
for i in xrange(1,100):
	print i
	path = np.array(sigs[i])
	imgO = imread(getFilePath(i),as_grey=True)
	# thresh = Filter.threshold_otsu(imgO)
	thresh = 0.9
	img = imgO < thresh
	y,x = np.nonzero(img)
	img = skeletonize(img)
	# y,x = np.nonzero(img)
	img = img[y.min():y.max(),x.min():x.max()]
	# imgO = imgO[y.min():y.max(),x.min():x.max()]
	sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
	summed = convolve(img,sumArr,mode='constant',cval=0)
	# corners = (((summed == 2) | (summed > 4)) & (img == 1))
	corners = ((summed == 2) & (img == 1))
	intersects = ((summed >= 4) & (img == 1))
	corners = np.transpose(np.array(np.nonzero(corners)))
	corners[:,1] += x.min()
	corners[:,0] += y.min()
	pathx = path[:,0]*img.shape[1]+x.min()
	pathy = path[:,1]*img.shape[0]+y.min()
	plt.subplot
	plt.imshow(img,cmap='gray')
	# plt.scatter(pathx,pathy,s=path[:,2])
	# plt.scatter(corners[:,1],corners[:,0],s=50,c='red')
	plt.show()
	startx = path[0][0]*img.shape[1]+x.min()
	starty = path[0][1]*img.shape[0]+y.min()
	start = np.transpose(np.array([[startx],[starty]]))
	corners[:,[0,1]] = corners[:,[1,0]]
	dists = cdist(start,corners)
	if dists.min() < 15:
		x,y = corners[dists.argmin()][0],corners[dists.argmin()][1]
		# corners = np.delete(corners,dists.argmin(),axis=0)
	else:
		x,y = int(startx),int(starty)
	# r = 100
	# correct = Neighbors(imgO,y-r/2,x-r/2,r).flatten()
	# wrongI = np.random.randint(0,corners.shape[0])
	# for i,corner in enumerate(corners):
	# 	if i % 4 == 0:
	# 		if dists[0][i] > 50:
	# 			wrongX,wrongY = corner[0],corner[1]
	# 			wrong = Neighbors(imgO,wrongY-r/2,wrongX-r/2,r).flatten()
	# 			ds.addSample(tuple(wrong),(0))
	# ds.addSample(tuple(correct),(1))
	# X.append(correct.flatten())
	# Y.append(1)
	# X.append(wrong.flatten())
	# Y.append(-1)
# net = buildNetwork(ds.indim, 100, ds.outdim, bias=True)
# t = BackpropTrainer(net, momentum=0.1, weightdecay=0.1, learningrate=0.01, lrdecay=0.999999, verbose = True)
# t.trainOnDataset(ds, 5)

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
# 	wrong = Neighbors(imgO,wrongY-r/2,wrongX-r/2,r)
# 	plt.subplot(1,2,1)
# 	plt.imshow(correct)
# 	plt.subplot(1,2,2)
# 	plt.imshow(wrong)
# 	correct = net.activate(correct.flatten())
# 	wrong = net.activate(wrong.flatten())
# 	print correct,wrong
# 	corrects.append(correct)
# 	wrongs.append(wrong)
# 	plt.show()
	# wGuess = lr.predict(wrong.flatten())
	# rGuess = lr.predict(correct.flatten())
	# if wGuess != -1:
	# 	errs.append(1)
	# else:
	# 	errs.append(0)
	# if rGuess != 1:
	# 	errs.append(1)
	# else:
	# 	errs.append(0)
	# print 'wrong',wGuess
	# print 'right',rGuess
# print np.mean(corrects)
# print np.mean(wrongs)
	
# lrx = linear_model.LogisticRegression()
# lry = linear_model.LogisticRegression()
# lrx.fit(X,Y[:,0])
# print 'fitted x'
# lry.fit(X,Y[:,1])
# print 'fitted Y'
# errsx = []
# errsy = []
# for i in xrange(500,600):
# 	print i
# 	x = transform(i)
# 	plt.imshow(x)
# 	path = np.array(sigs[i])
# 	yx = path[0][0]
# 	yy = path[0][1]
# 	yx = np.abs(lrx.predict(x.flatten())-yx)
# 	yy = np.abs(lry.predict(x.flatten())-yy)
# 	plt.scatter(yx*100,(1-yy)*100,c='red')
# 	plt.show()
# 	errsx.append(yx)
# 	errsy.append(yy)
# errsx = np.array(errsx)
# errsy = np.array(errsy)
# plt.hist(errsx,bins=20)
# print errsx.sum()/errsx.size
# print errsy.sum()/errsy.size
# plt.show()
# plt.hist(errsy,bins=20)
# plt.show()

	# plt.subplot(1,3,1)
	# plt.scatter(path[:,0],1-path[:,1],s=path[:,2])
	# plt.subplot(1,3,2)
	# plt.imshow(imgO,cmap='gray')
	# plt.subplot(1,3,3)
	# plt.imshow(img,cmap='gray')
	# plt.show()
# for i in xrange(len(sigs),1081):
# 	print i
# 	i = transform(i)
# 	i = i < 0.9
# 	i = skeletonize(i)
# 	points = np.transpose(np.array(np.nonzero(i)))
# 	imgErrs = []
# 	for img in imgs:
# 		errs =[]
# 		for point in points:
# 			point = np.transpose(np.array([[point[0]],[point[1]]]))
# 			dists = cdist(point,img)
# 			errs.append(dists.min())
# 		imgErrs.append(np.sum(errs))
# 	plt.plot(imgErrs)
# 	plt.show()
	# errs = []
	# for j in imgs:
	# 	errs.append(np.abs(i-j).sum())
	# plt.plot(errs)
	# plt.show()
