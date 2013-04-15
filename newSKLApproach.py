import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from time import time,sleep
from random import randint
from PIL import Image
import cPickle
import os

from skimage.io import imread
from skimage.exposure import equalize_hist
import skimage.filter as Filter
from skimage.morphology import skeletonize,medial_axis,selem,binary_erosion
from skimage.transform import resize
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_foerstner
from skimage.measure import find_contours,structural_similarity,regionprops
from skimage.segmentation import felzenszwalb,slic,quickshift,mark_boundaries
from skimage.transform import hough, hough_peaks, probabilistic_hough

from scipy.spatial.distance import cdist
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import convolve,gaussian_filter
from scipy import interpolate
from scipy.interpolate import Rbf

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.feature_selection import RFE

def Neighbors(arr,x,y,n=3):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    return arr[:n,:n]

def getFilePath(n):
	base = 'images/'
	filepath = base+('0'*(4-len(str(n))))+str(n)+'.jpg'
	return filepath

def transform(n,rsizeP,padP,justImg=True):
	imgO = imread(getFilePath(n),as_grey=True)
	# thresh = Filter.threshold_otsu(imgO)
	thresh = 0.9
	img = imgO < thresh
	# img = skeletonize(img)
	y,x = np.nonzero(img)
	imgO = imgO[y.min():y.max(),x.min():x.max()]
	imgR = resize(imgO,(rsizeP,rsizeP))
	imgP = np.pad(imgR,(padP,padP),mode='constant',constant_values=1)
	if justImg:
		return imgR
	else:
		return imgR,imgP,y.max()-y.min(),x.max()-x.min()

def getErr(target,predicted):
	errX = np.sqrt(np.sum(np.square(predicted[:,0]-target[:,0]))/len(target))
	errY = np.sqrt(np.sum(np.square(predicted[:,1]-target[:,1]))/len(target))
	return (errX+errY)/2

def genPredictions(x,y,target):
	predicted = np.ones_like(target[:,:2])
	predicted[:,0] = predicted[:,0]*y
	predicted[:,1] = predicted[:,1]*x
	return predicted

def tCoords(n):
	imgO = imread(getFilePath(n),as_grey=True)
	sig = np.array(sigs[n])
	thresh = 0.75
	img = imgO < thresh
	x,y = np.nonzero(img)
	xR = x.max()-x.min()
	yR = y.max()-y.min()
	xA = np.mean(x)
	yA = np.mean(y)
	xGuess = (xA-x.min())/xR
	yGuess = (yA-y.min())/yR
	# return xGuess,yGuess
	return xGuess,yGuess

def tPredictions(n,target):
	xGuess,yGuess = tCoords(n)
	return genPredictions(xGuess,yGuess,target)

def genPredictions2Points(x,y,x2,y2,target):
	predicted = np.ones_like(target[:,:2])
	half = target.shape[0]/2.0
	predicted[:,0][target[:,2] <= half] = predicted[:,0][target[:,2] <= half]*y
	predicted[:,1][target[:,2] <= half] = predicted[:,1][target[:,2] <= half]*x
	predicted[:,0][target[:,2] > half] = predicted[:,0][target[:,2] > half]*y2
	predicted[:,1][target[:,2] > half] = predicted[:,1][target[:,2] > half]*x2
	return predicted

def getImgData(img):
	img = img.reshape(100,100)
	labelPos = np.array((img < 0.9),dtype=int)
	# labelNeg = np.array((img >= 0.9),dtype=int)
	props = ['Area', 'Centroid','WeightedCentroid','WeightedMoments','WeightedHuMoments','HuMoments','EulerNumber','Eccentricity','EquivDiameter','Extent','MeanIntensity','MinIntensity','MaxIntensity']
	# props = ['Centroid','WeightedCentroid']
	dataPos = regionprops(labelPos,properties=props,intensity_image=img)[0]
	del dataPos['Label']
	# dataNeg = regionprops(labelNeg,properties=props,intensity_image=img)[0]
	# del dataNeg['Label']
	return dataPos#,dataNeg

def makeFeatVect(img):
	data = getImgData(img)
	featVect = []
	for datum in data.values():
		featVect.extend(np.array(datum).flatten())
	return np.array(featVect)

def MLPredict(n):
	img = transform(n,100,0,True).flatten()
	x = makeFeatVect(img)

def genSKIMGPredictions(n):
	imgO = imread(getFilePath(n),as_grey=True)
	thresh = 0.75
	img = imgO < thresh
	x,y = np.nonzero(img)
	xR = x.max()-x.min()
	yR = y.max()-y.min()
	labelPos = np.array((img < 0.8),dtype=int)
	props = ['WeightedCentroid']
	dataPos = regionprops(labelPos,properties=props,intensity_image=img)[0]
	print dataPos
	# xGuess = (xA-x.min())/xR
	# yGuess = (yA-y.min())/yR

minPos = cPickle.load(open('usefulData/minPos.pkl','rb'))
minPos2Ps = cPickle.load(open('usefulData/minPos2Points.pkl','rb'))
tPos = cPickle.load(open('usefulData/tPos.pkl','rb'))
minErrs = cPickle.load(open('usefulData/minErrs.pkl','rb'))
tErrs = cPickle.load(open('usefulData/tErrs.pkl','rb'))
imgs = cPickle.load(open('usefulData/imgs100px.pkl'))

# imgs = cPickle.load(open('usefulData/imgs200px.pkl'))
# imgs = cPickle.load(open('usefulData/imgs100pxFull.pkl'))

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

imgData = []
imgNums = []
for i,img in enumerate(imgs):
	if i % 50 == 0: print i
	imgData.append(makeFeatVect(imgs[i]))
# for i in xrange(1,len(sigs)+1):
# 	if i % 50 == 0: print i
# 	imgData.append(tCoords(i))
# imgData = np.array(imgData)
minPos = []
for i in xrange(1,len(sigs)+1):
	x,y,t = np.mean(sigs[i],axis=0)
	minPos.append([x,y])
# 	# plt.imshow(imgs[i-1].reshape(100,100),cmap='gray')
# 	# plt.scatter(x*100,y*100)
# 	# plt.show()
minPos = np.array(minPos)
# plt.scatter(minPos[:,0],minPos[:,1],c='red')
# plt.scatter(imgData[:,0],imgData[:,1],c='blue')
# plt.show()
# minPos = np.array(minPos)
# cPickle.dump(minPos,open('minPos.pkl','wb'))

# trainX = np.array(imgs[:605])
# trainY = np.array(minPos)
# testX = np.array(imgs[606:])
# trainX = np.array(imgData[:605])
# trainY = np.array(minPos)
# testX = np.array(imgData[606:])
# # plt.imshow(testX[0].reshape(100,100),cmap='gray')
# # plt.show()

# trainX = np.array(imgs[:500])
# trainY = np.array(minPos[:500])
# testX = np.array(imgs[500:])
# testY = np.array(minPos[500:])

print len(imgData)
print len(minPos)
nTrain = 500
trainX = np.array(imgData[:nTrain])
trainY = np.array(minPos[:nTrain])
testX = np.array(imgData[nTrain:])
testY = np.array(minPos[nTrain:])
# trainX = np.array(imgData[100:600])
# trainY = np.array(minPos[100:600])
# testX = np.array(imgData[:100])
# testY = np.array(minPos[:100])

scaler = preprocessing.StandardScaler().fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

# trainX = np.array(imgs[:500])
# trainY = np.array(minPos2Ps[:500])
# testX = np.array(imgs[500:])
# testY = np.array(minPos2Ps[500:-1])

print trainX.shape
print trainY.shape
print testX.shape
print testY.shape

# clf = GradientBoostingRegressor()
# clf = linear_model.lasso_path()
# clfX = GradientBoostingRegressor(verbose=2)
# clfY = GradientBoostingRegressor(verbose=2)
# clf = linear_model.Ridge()
# clf = RandomForestRegressor(n_estimators=100,n_jobs=4,verbose=2)
# clf2 = RandomForestRegressor(n_estimators=100,n_jobs=4,verbose=2)
# parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':np.linspace(1,10,10), 'epsilon':np.logspace(-3,2,10)}
parameters = {'kernel':('poly', 'rbf'), 'C':[1,10], 'epsilon':[0,1]}
svr1 = svm.SVR()
svr2 = svm.SVR()
clfX = GridSearchCV(svr1, parameters)
clfY = GridSearchCV(svr2, parameters)
# clfX = RFE(svm.SVR(kernel="linear",verbose=2), 32, step=1)
# clfY = RFE(svm.SVR(kernel="linear",verbose=2), 32, step=1)
# clfX = RFE(svm.SVR(kernel="linear",verbose=2), 32, step=1)
# clfY = RFE(svm.SVR(kernel="linear",verbose=2), 32, step=1)
# clfX = svm.SVR(verbose=2)
# clfY = svm.SVR(verbose=2)

clfX.fit(trainX,trainY[:,0])
clfY.fit(trainX,trainY[:,1])

# print clfX

predictionsX = clfX.predict(testX)
predictionsY = clfY.predict(testX)
predictions = np.vstack((predictionsX,predictionsY)).T

# tX = clfX.predict(scaler.transform(makeFeatVect(imgs[600])))
# tY = clfY.predict(scaler.transform(makeFeatVect(imgs[600])))
# print tX,tY
# plt.imshow(imgs[550].reshape(100,100),cmap='gray')
# plt.scatter(tY*100,tX*100)
# plt.scatter(testY[550-500][0]*100,testY[550-500][1]*100,c='red')
# mX,mY,mT = np.mean(sigs[550],axis=0)
# plt.scatter(mX*100,mY*100,c='green')
# plt.show()
# clf.fit(trainX,trainY)
# clf2.fit(trainX,trainY)
# predictions = clf.predict(testX)

pXs = predictions[:,0]
pYs = predictions[:,1]
print pXs[0],pYs[0]

# pX2s = predictions[:,2]
# pY2s = predictions[:,3]

# img = testX[0].reshape(100,100)
# path = np.array(sigs[501])
# plt.imshow(img,cmap='gray')
# plt.scatter(path[:,0]*100,path[:,1]*100)
# plt.show()

errsT = []
errsM = []
errsML = []
errs05 = []
for i in xrange(len(pXs)):
	sig = np.array(sigs[nTrain+i+1])
	MLPredictions = genPredictions(pXs[i],pYs[i],sig)
	# SKIMGPredictions = genSKIMGPredictions(i+1)
	# predictions = genPredictions2Points(pXs[i],pYs[i],pX2s[i],pY2s[i],sig)
	errML = getErr(sig,MLPredictions)
	zeropointfives = np.ones_like(MLPredictions)/2.0
	err05 = getErr(sig,zeropointfives)
	mX = minPos[nTrain+i][0]
	mY = minPos[nTrain+i][1]
	mPredictions = genPredictions(mX,mY,sig)
	errM = getErr(sig,mPredictions)
	errT = getErr(sig,tPredictions(nTrain+i+1,sig))
	errsT.append(errT)
	errsM.append(errM)
	errsML.append(errML)
	errs05.append(err05)
print 'ml err',np.mean(errsML)
print '0.5s err',np.mean(errs05)
print 'threshold err',np.mean(errsT)
print 'min err',np.mean(errsM)

# errsM = np.ones_like(predictions)/2.0

# print 'ml approach err',np.sum(np.abs(predictions-testY))/len(testY)
# print 'threshold approach err',np.sum(np.abs(tPos[500:-1]-testY))/len(testY)
# print '0.5s err',np.sum(np.abs(zeropointfives-testY))/len(testY)

# reader = open('testTest.csv')
# outWriter = csv.writer(open('testTestSubmission.csv','wb'))
# for i,line in enumerate(reader):
# 	if i == 0:
# 		outWriter.writerow(line.split(','))
# 	if i > 0:
# 		if i % 10000 == 0: print i
# 		data = [field.strip() for field in line.split(',')]
# 		idx = int(data[1])
# 		try:
# 			data[5] = pXs[idx-606]
# 			data[6] = pYs[idx-606]
# 			print data[5],data[6]
# 			sleep(1)
# 		except:
# 			data[5] = 0.5
# 			data[6] = 0.5
# 		outWriter.writerow(data)

# reader = open('testTest.csv')
# outWriter = csv.writer(open('testTestSubmission.csv','wb'))
# curImg = 0
# for i,line in enumerate(reader):
# 	if i == 0:
# 		outWriter.writerow(line.split(','))
# 	if i > 0:
# 		if i % 10000 == 0: print i
# 		data = [field.strip() for field in line.split(',')]
# 		imgNum = int(data[1])
# 		if imgNum != curImg:
# 			curImg = imgNum
# 			curX,curY = MLPredict(curImg)
# 		try:
# 			data[5] = pXs[idx-606]
# 			data[6] = pYs[idx-606]
# 			print data[5],data[6]
# 			sleep(1)
# 		except:
# 			data[5] = 0.5
# 			data[6] = 0.5
# 		outWriter.writerow(data)