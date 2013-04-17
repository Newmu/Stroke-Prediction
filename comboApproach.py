import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from time import time,sleep
from random import randint
from PIL import Image
import cPickle
import os
import sys, traceback

from skimage.io import imread
from skimage.exposure import equalize_hist
import skimage.filter as Filter
from skimage.morphology import skeletonize,medial_axis,selem,binary_erosion,disk
from skimage.transform import resize
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_foerstner
from skimage.measure import find_contours,structural_similarity,regionprops
from skimage.segmentation import felzenszwalb,slic,quickshift,mark_boundaries
from skimage.transform import hough, hough_peaks, probabilistic_hough
from skimage.filter.rank import entropy

from scipy.spatial.distance import cdist
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import convolve,gaussian_filter
from scipy import interpolate
from scipy.interpolate import Rbf
from scipy.stats import linregress,skew

def Neighbors(arr,x,y,n=3):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    return arr[:n,:n]

def getFilePath(n):
	base = 'images/'
	filepath = base+('0'*(4-len(str(n))))+str(n)+'.jpg'
	return filepath

def getErr(target,predicted):
	errX = np.sqrt(np.sum(np.square(predicted[:,0]-target[:,0]))/len(target))
	errY = np.sqrt(np.sum(np.square(predicted[:,1]-target[:,1]))/len(target))
	return (errX+errY)/2

def genPredictions(x,y,target):
	predicted = np.ones_like(target[:,:2])
	predicted[:,0] = predicted[:,0]*y
	predicted[:,1] = predicted[:,1]*x
	return predicted

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
		if padP == 0:
			return imgR
		else:
			return imgR,imgP
	else:
		return imgR,imgP,y.max()-y.min(),x.max()-x.min()

def mean_curvature(Z):
    Zy, Zx  = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, Zyx = np.gradient(Zy)
    H = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx
    H = -H/(2*(Zx**2 + Zy**2 + 1)**(1.5))
    return H

def getCurvs(img,nBoxes,nPadded,winSize):
	xStrides = np.linspace(nPadded,img.shape[0]-nPadded,nBoxes).astype(int)
	yStrides = np.linspace(nPadded,img.shape[0]-nPadded,nBoxes).astype(int)
	# print nPadded,img.shape[0]-nPadded,img.shape[0]-nPadded*2,xStrides
	# plt.imshow(img,cmap='gray')
	# plt.show()
	curvs = []
	for x in xStrides:
		for y in yStrides:
			center = int(winSize/2)
			imgWin = Neighbors(img,x-nPadded,y-nPadded,winSize)
			if imgWin[center,center] != 0:
				segs,nSegs = label(imgWin)
				segLabel = segs[center,center]
				segs[segs != segLabel] = 0
				xs,ys = np.nonzero(segs)
				# xs,ys = np.nonzero(imgWin)
				# plt.subplot(1,2,1)
				# plt.imshow(imgWin,cmap='gray')
				# plt.subplot(1,2,2)
				# plt.imshow(segs)
				# plt.show()
				# residErr = linregress(xs,ys)[4]
				# residErr = np.abs(np.polyfit(xs,ys,deg=1)[1])/float(len(xs))
				# residErr = np.abs(np.polyfit(xs,ys,deg=2)[0])
				xErr = np.linalg.lstsq(np.vstack([xs, np.ones(len(xs))]).T,ys)[1]/float(xs.size)
				yErr = np.linalg.lstsq(np.vstack([ys, np.ones(len(xs))]).T,xs)[1]/float(ys.size)
				# residErr = np.sqrt((xErr+yErr)/2.0)
				residErr = (xErr+yErr)/2.0
				if residErr.size == 0:
					residErr = np.mean(curvs)
				# print residErr
				# cCoefs = np.abs(np.corrcoef(xs,ys))
				# xVal = skew(xs)
				# yVal = skew(ys)
				# residErr = (xVal+yVal)/2
				curvs.append(residErr)
				# if cCoefs != []:
					# curvs.append(1-cCoefs[1,0])
				# else:
					# curvs.append(0)
			else:
				curvs.append(0)
			# plt.imshow(imgWin,cmap='gray')
			# plt.show()
	curvs = np.array(curvs)
	curvs = np.nan_to_num(curvs)
	return curvs

# def getCurvs(img,nBoxes,nPadded,winSize):
# 	xStrides = np.linspace(nPadded,img.shape[0]-nPadded,nBoxes).astype(int)
# 	yStrides = np.linspace(nPadded,img.shape[0]-nPadded,nBoxes).astype(int)
# 	# print nPadded,img.shape[0]-nPadded,img.shape[0]-nPadded*2,xStrides
# 	# plt.imshow(img,cmap='gray')
# 	# plt.show()
# 	curvs = []
# 	for x in xStrides:
# 		for y in yStrides:
# 			imgWin = Neighbors(img,x-nPadded,y-nPadded,winSize)
# 			# segs,nSegs = label(imgWin)
# 			# plt.subplot(1,2,1)
# 			# plt.imshow(imgWin,cmap='gray')
# 			# plt.subplot(1,2,2)
# 			# plt.imshow(segs)
# 			# plt.show()
# 			xs,ys = np.nonzero(imgWin)
# 			cCoefs = np.corrcoef(xs,ys)
# 			if cCoefs != []:
# 				curvs.append(1-np.abs(np.corrcoef(xs,ys)[1,0]))
# 			else:
# 				curvs.append(0)
# 			# plt.imshow(imgWin,cmap='gray')
# 			# plt.show()
# 	curvs = np.array(curvs)
# 	curvs = np.nan_to_num(curvs)
# 	return curvs

def weightedMeanPos(img,weights):
	x,y = np.nonzero(img)
	xR = x.max()-x.min()
	yR = y.max()-y.min()
	xA = np.average(x,weights=weights)
	yA = np.average(y,weights=weights)
	xAvg = (xA-x.min())/xR
	yAvg = (yA-y.min())/yR
	return xAvg,yAvg

def meanPos(img):
	x,y = np.nonzero(img)
	xR = x.max()-x.min()
	yR = y.max()-y.min()
	xA = np.mean(x)
	yA = np.mean(y)
	xMean = (xA-x.min())/xR
	yMean = (yA-y.min())/yR
	return xMean,yMean

def Predict(n):
	try:
		rawImg = imread(getFilePath(n),as_grey=True)
		# target = np.array(sigs[n])

		rawImg = gaussian_filter(rawImg,2)
		img = skeletonize(rawImg < 0.95)
		y,x = np.nonzero(img)
		img = img[y.min():y.max(),x.min():x.max()]
		sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
		summed = convolve(img,sumArr,mode='constant',cval=0)
		# corners = ((summed == 2) & (img == 1))
		corners = ((summed > 3) & (img == 1))
		corners = np.transpose(np.array(np.nonzero(corners)))
		xR = x.max()-x.min()
		yR = y.max()-y.min()
		cTX = np.mean(corners[:,0])/yR
		cTY = np.mean(corners[:,1])/xR
		# CTErr = getErr(genPredictions(cTX,cTY,target),target)
		# CTErrs.append(CTErr)

		# # plt.subplot(1,2,1)
		# plt.imshow(rawImg[y.min():y.max(),x.min():x.max()],cmap='gray')
		# plt.scatter(corners[:,1],corners[:,0])
		# plt.show()
		# print corners.shape
		# plt.subplot(1,2,1)
		# plt.imshow(rawImg < 0.9)
		# plt.subplot(1,2,2)
		# plt.imshow(entropy((rawImg*255).astype(np.uint16), disk(5)))
		# plt.show()
		# corners = corner_peaks(corner_harris(rawImg))
		# contours = find_contours(rawImg, thres)
		# plt.imshow(rawImg,cmap='gray')
		# for contour in contours:
		# 	plt.plot(contour[:,1],contour[:,0])
		# plt.show()

		rImg,pImg = transform(n,200,padSize,True)
		rawBinary = np.array((rawImg < thres),dtype=int)
		pBinary = np.array((pImg < thres),dtype=int)
		rBinary = np.array((rImg < thres),dtype=int)

		tX,tY = meanPos(rawBinary)
		# NWPs = genPredictions(tX,tY,target)
		# tErr = getErr(NWPs,target)
		# tErrs.append(tErr)

		weights = getCurvs(pBinary,nBoxes,padSize,curvWinSize)
		# weights = weights+1
		weights = weights.max()/5+weights
		# plt.subplot(1,2,1)
		# plt.imshow(rImg)
		# plt.subplot(1,2,2)
		# plt.imshow(weights.reshape(nBoxes,nBoxes),cmap='gray')
		# plt.show()
		weights = weights[(rImg.flatten() < thres)]
		wX,wY = weightedMeanPos(rBinary,weights)
		# WPs = genPredictions(wX,wY,target)
		# cErr = getErr(WPs,target)
		# cErrs.append(cErr)

		# weights = 1-rImg[rImg < thres]
		# iX,iY = weightedMeanPos(rBinary,weights)
		# IPs = genPredictions(iX,iY,target)
		# iErr = getErr(IPs,target)
		# iErrs.append(iErr)
		pX = (wX*3+tX*4+cTX)/8
		pY = (wY*3+tY*4+cTY)/8
		return pX,pY
		# APs = genPredictions(pX,pY,target)
		# aErr = getErr(APs,target)
		# aErrs.append(aErr)
	except:
		print 'something went wrong at',n
		print traceback.print_exc(file=sys.stdout)
		return 0.5,0.5

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

reader = open('test.csv')
tests = defaultdict(list)
for i,line in enumerate(reader):
	if i > 0:
		data = [field.strip() for field in line.split(',')]
		fields = ['row','sig','writer','occurence','time','x','y']
		data = dict(zip(fields,data))
		tests[int(data['sig'])].append([0,0,float(data['time'])])

print 'loaded all data in',time()-t0

# nBoxes = 201
# winSize = 20
# padSize = int(winSize/2.0)
# for i in xrange(100):
# 	i = randint(1,len(sigs)+1)
# 	print i
# 	img = transform(i,200,padSize,True)
# 	print img.shape
# 	binary = np.array((img < 0.9),dtype=int)
# 	skel = skeletonize(binary)
# 	curvs = getCurvs(binary,nBoxes,padSize,9)
# 	print curvs
# 	plt.subplot(2,2,1)
# 	plt.imshow(img,cmap='gray')
# 	plt.subplot(2,2,2)
# 	plt.imshow(curvs.reshape(nBoxes,nBoxes),cmap='gray')
# 	plt.subplot(2,2,3)
# 	plt.hist(curvs,bins=50)
# 	curvature = mean_curvature(binary)
# 	plt.subplot(2,2,4)
# 	plt.imshow(curvature,cmap='gray')
# 	plt.show()
# plt.subplot(1,2,1)
# plt.imshow(img,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(binary,cmap='gray')
# plt.show()

nBoxes = 201
thres = 0.80
curvWinSize = 15
padSize = int(curvWinSize/2.0)
print curvWinSize,thres,padSize
print 'new curv function, no stroke elims'
tErrs = []
cErrs = []
iErrs = []
aErrs = []
CTErrs = []
for i in xrange(1,len(sigs)+1):
	i = randint(1,len(sigs)+1)
	Predict(i)

# errsT = []
# errsM = []
# errsML = []
# errs05 = []
# for i in xrange(len(pXs)):
# 	sig = np.array(sigs[nTrain+i+1])
# 	errML = getErr(sig,MLPredictions)
# 	zeropointfives = np.ones_like(MLPredictions)/2.0
# 	err05 = getErr(sig,zeropointfives)
# 	mX = minPos[nTrain+i][0]
# 	mY = minPos[nTrain+i][1]
# 	mPredictions = genPredictions(mX,mY,sig)
# 	errM = getErr(sig,mPredictions)
# 	errT = getErr(sig,tPredictions(nTrain+i+1,sig))
# 	errsT.append(errT)
# 	errsM.append(errM)
# 	errsML.append(errML)
# 	errs05.append(err05)
# print '0.5s err',np.mean(errs05)
# print 'threshold err',np.mean(errsT)
# print 'min err',np.mean(errsM)

reader = open('testTest.csv')
outWriter = csv.writer(open('YXSubmission.csv','wb'))
outWriter2 = csv.writer(open('XYSubmission.csv','wb'))
curImg = 0
for i,line in enumerate(reader):
	if i == 0:
		outWriter.writerow(line.split(','))
	if i > 0:
		if i % 10000 == 0: print i
		data = [field.strip() for field in line.split(',')]
		imgNum = int(data[1])
		if imgNum != curImg:
			print 'loading img',imgNum
			curImg = imgNum
			curX,curY = Predict(curImg)
			print curX,curY
		data[5] = curX
		data[6] = curY
		outWriter.writerow(data)
		data[5] = curY
		data[6] = curX
		outWriter2.writerow(data)