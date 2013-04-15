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
from skimage.exposure import equalize_hist
import skimage.filter as Filter
from skimage.morphology import skeletonize,medial_axis,selem,binary_erosion
from skimage.transform import resize
from skimage.feature import corner_harris, corner_subpix, corner_peaks, corner_foerstner
from skimage.measure import find_contours,structural_similarity
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

def interpPath(path,n):
	t = np.copy(path[:,2])
	x = np.copy(path[:,0])
	y = np.copy(path[:,1])
	mask = ((np.diff(x) == 0) & (np.diff(y) == 0))
	t[mask] += np.random.random(len(t[mask]))/10000.0
	x[mask] += np.random.random(len(t[mask]))/100.0
	y[mask] += np.random.random(len(t[mask]))/100.0
	tck,u = interpolate.splprep([x,y],k=1,s=0)
	tnew = np.linspace(0,1,n)
	outx,outy = interpolate.splev(tnew,tck)
	return np.vstack((outx,outy,tnew)).T

def calcErr(actual,predicted):
	errX = 0
	errY = 0
	errX += np.sqrt(np.sum(np.square(predicted[:,0]-actual[:,0]))/len(actual))
	errY += np.sqrt(np.sum(np.square(predicted[:,1]-actual[:,1]))/len(actual))
	return (errX+errY)/2

os.system("taskset -p 0xff %d" % os.getpid())
np.set_printoptions(suppress=True)

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

# tSims = []
# for i in xrange(1,100):
# 	# img1 = imread(getFilePath(i),as_grey=True)
# 	# img2 = imread(getFilePath(i+1),as_grey=True)
# 	img1 = transform(i,300,0)
# 	sims = []
# 	for j in xrange(1,600):
# 		img2 = transform(j,300,0)
# 		sim = structural_similarity(img1,img2)
# 		print sim
# 		sims.append(sim)
# 	plt.plot(sims)
# 	plt.show()
# 	sims = np.array(sims)
# 	tSims.append([sims.argmin(),sims.min()])

# for i in xrange(1,len(sigs)+1):
# 	img = imread(getFilePath(i),as_grey=True)
# 	img[img > 0.95] = 0.95
# 	imgEq = equalize_hist(img)
# 	binary = imgEq < 0.9
# 	imgX,imgY = np.nonzero(binary)
# 	imgW = imgX.max()-imgX.min()
# 	imgH = imgY.max()-imgY.min()
# 	binary = binary[imgX.min():imgX.max(),imgY.min():imgY.max()]
# 	# skel = skeletonize(binary)
# 	skel,dist = medial_axis(binary, return_distance=True)
# 	dist_on_skel = dist * skel
# 	# plt.subplot(1,2,1)
# 	# plt.imshow(binary)
# 	# plt.subplot(1,2,2)
# 	# plt.imshow(binary_erosion(binary,selem.diamond(2)))
# 	# img = imread(getFilePath(42))
# 	# segs = felzenszwalb(img, scale=100, sigma=1, min_size=50)
# 	segs = slic(img, ratio=0.001, n_segments=2, sigma=1)
# 	# segs = quickshift(img, kernel_size=3, max_dist=6, ratio=10)
# 	plt.imshow(segs)
# 	plt.show()
# 	path = np.array(sigs[1])
# 	print path.shape
# 	path2 = np.array(sigs[42])
# 	# print len(path)
# 	# print path
# 	interped = interpPath(path2,len(path))
# 	print interped.shape
# 	print calcErr(path,interped)
# 	# interped = interpPath(path,150)
# 	# print interped.shape
# 	# plt.imshow(img,cmap='gray')
# 	# plt.plot(path[:,0],path[:,1],'r')
# 	# plt.plot(interped[:,0],interped[:,1],'b')
# 	plt.show()

# sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
# for i in xrange(1,len(sigs)+1):
# 	# i = randint(1,len(sigs)+1)
# 	# i = 1
# 	print i
# 	img = imread(getFilePath(i),as_grey=True)
# 	img = gaussian_filter(img,2)
# 	path = np.array(sigs[i])
# 	img[img > 0.95] = 0.95
# 	binary = img < 0.9
# 	imgX,imgY = np.nonzero(binary)
# 	imgW = imgX.max()-imgX.min()
# 	imgH = imgY.max()-imgY.min()
# 	binary = binary[imgX.min():imgX.max(),imgY.min():imgY.max()]
# 	img = img[imgX.min():imgX.max(),imgY.min():imgY.max()]
# 	h, theta, d = hough(binary)
# 	rows, cols = binary.shape
# 	plt.imshow(img,cmap='gray')
# 	for _, angle, dist in zip(*hough_peaks(h, theta, d,min_angle=50,threshold=h.max()*0.4)):
# 	    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
# 	    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
# 	    plt.plot((0, cols), (y0, y1), '-r')
# 	plt.show()
# 	skel = skeletonize(binary)
# 	summed = convolve(skel,sumArr,mode='constant',cval=0)
# 	corners = ((summed == 2) & (skel == 1))
# 	corners = np.transpose(np.array(np.nonzero(corners)))
# 	startx = path[0][0]*binary.shape[1]
# 	starty = path[0][1]*binary.shape[0]
# 	img = img[imgX.min():imgX.max(),imgY.min():imgY.max()]
# 	img[skel != 0] = 0.1
# 	plt.imshow(img,cmap='gray')
# 	plt.scatter(startx,starty,c='red')
# 	plt.scatter(corners[:,1],corners[:,0],c='green')
# 	plt.show()

starts = []
one = []
two = []
three = []
four = []
five = []
six = []
seven = []
eight = []
nine = []
ten = []
for i in xrange(1,len(sigs)+1):
	path = np.array(sigs[i])
	path[:,2] = path[:,2]/float(path[:,2].max())
	one.extend(path[(path[:,2] <= 0.1)])
	two.extend(path[(path[:,2] >= 0.1) & (path[:,2] < 0.2)])
	three.extend(path[(path[:,2] >= 0.2) & (path[:,2] < 0.3)])
	four.extend(path[(path[:,2] >= 0.3) & (path[:,2] < 0.4)])
	five.extend(path[(path[:,2] >= 0.4) & (path[:,2] < 0.5)])
	six.extend(path[(path[:,2] >= 0.5) & (path[:,2] < 0.6)])
	seven.extend(path[(path[:,2] >= 0.6) & (path[:,2] < 0.7)])
	eight.extend(path[(path[:,2] >= 0.7) & (path[:,2] < 0.8)])
	nine.extend(path[(path[:,2] >= 0.8) & (path[:,2] < 0.9)])
	ten.extend(path[(path[:,2] >= 0.9) & (path[:,2] < 1.0)])
	starts.extend(path[0])
one = np.mean(one,axis=0)
two = np.mean(two,axis=0)
three = np.mean(three,axis=0)
four = np.mean(four,axis=0)
five = np.mean(five,axis=0)
six = np.mean(six,axis=0)
seven = np.mean(seven,axis=0)
eight = np.mean(eight,axis=0)
nine = np.mean(nine,axis=0)
ten = np.mean(ten,axis=0)
print one
plt.scatter(one[0],one[1],s=1)
plt.scatter(two[0],two[1],s=2)
plt.scatter(three[0],three[1],s=3)
plt.scatter(four[0],four[1],s=4)
plt.scatter(five[0],five[1],s=5)
plt.scatter(six[0],six[1],s=6)
plt.scatter(seven[0],seven[1],s=7)
plt.scatter(eight[0],eight[1],s=8)
plt.scatter(nine[0],nine[1],s=9)
plt.scatter(ten[0],ten[1],s=10)
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()
for i in xrange(1,len(sigs)+1):
	path = np.array(sigs[i])
	img = imread(getFilePath(i),as_grey=True)
	img[img > 0.95] = 0.95
	binary = img < 0.9
	imgX,imgY = np.nonzero(binary)
	imgW = imgX.max()-imgX.min()
	imgH = imgY.max()-imgY.min()
	binary = binary[imgX.min():imgX.max(),imgY.min():imgY.max()]
	img = img[imgX.min():imgX.max(),imgY.min():imgY.max()]
	plt.imshow(img,cmap='gray')
	plt.scatter(path[:,0]*imgH,path[:,1]*imgW,c='red')
	plt.show()
	starts.append(path[0])
starts = np.array(starts)
plt.scatter(starts[:,0],starts[:,1])
plt.show()

# ti = np.linspace(0, 1, 100)
# XI, YI = np.meshgrid(ti, ti)
# vals = np.ones(len(starts))
# print vals.shape
# print starts.shape
# rbf = Rbf(starts[:,0], starts[:,1], vals, epsilon=0.1)
# ZI = rbf(XI, YI)
# plt.pcolor(XI, YI, ZI, cmap='gray')
# plt.scatter(starts[:,0],starts[:,1])
# plt.show()