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
from PIL import Image
from rbm import rbm
import cPickle
from scipy import interpolate

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

def pilResize(a,size):
	pilImg = Image.fromarray(a)
	pilImg = pilImg.resize(size, Image.ANTIALIAS)
	return np.array(pilImg)


reader = open('train.csv')
sigs = defaultdict(list)
for i,line in enumerate(reader):
	if i > 0:
		data = [field.strip() for field in line.split(',')]
		fields = ['row','sig','writer','occurence','time','x','y']
		data = dict(zip(fields,data))
		sigs[int(data['sig'])].append([float(data['x']),float(data['y']),float(data['time'])])

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

imgs = []
paths = []
# # for i in xrange(1,len(sigs)):
for i in xrange(1,len(sigs)):
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
	imgO = imgO[y.min():y.max(),x.min():x.max()]
	sumArr = np.array([[1,1,1],[1,1,1],[1,1,1]])
	summed = convolve(img,sumArr,mode='constant',cval=0)
	# corners = (((summed == 2) | (summed > 4)) & (img == 1))
	corners = ((summed == 2) & (img == 1))
	intersects = ((summed >= 4) & (img == 1))
	corners = np.transpose(np.array(np.nonzero(corners)))
	# corners[:,1] += x.min()
	# corners[:,0] += y.min()
	# img01 = pilResize(imgO,(100,100))
	# img02 = resize(imgO,(100,100))
	# img01 = np.array((img01 > thresh),dtype=int)
	# img02 = np.array((img02 > thresh),dtype=int)
	path[:,0] = path[:,0]*imgO.shape[1]
	path[:,1] = path[:,1]*imgO.shape[0]
	path[:,2] = path[:,2]/path[:,2].max()
	t = np.copy(path[:,2])
	x = np.copy(path[:,0])
	y = np.copy(path[:,1])
	mask = ((np.diff(x) == 0) & (np.diff(y) == 0))
	t[mask] += np.random.random(len(t[mask]))/10000.0
	x[mask] += np.random.random(len(t[mask]))/100.0
	y[mask] += np.random.random(len(t[mask]))/100.0
	tck,u = interpolate.splprep([x,y],k=1,s=0.1)
	tnew = np.linspace(0,1,300)
	outx,outy = interpolate.splev(tnew,tck)
	# tck,u = interpolate.bisplrep(x,y,t)
	# unew = np.linspace(0,1,300)
	# outx,outy = interpolate.bisplev(unew,tck)
	# print outx
	# print outy
	# plt.imshow(imgO,cmap='gray')
	# plt.plot(path[:,0],path[:,1],'r')
	# plt.plot(outx,outy,'b')
	paths.append([outx,outy,tnew])
	# plt.scatter(corners[:,1],corners[:,0],s=50,c='red')
	# imgs.append(img02.flatten())
	# plt.show()
paths = np.array(paths)
# imgs = np.array(imgs)
# imgs = cPickle.load(open('imgs100pxFull.p','rb'))
# print imgs.shape
# cPickle.dump(imgs,open('imgs100pxFull.p','wb'))
cPickle.dump(paths,open('pathsLinInterp.p','wb'))

# rbm = rbm(10000,1000,0.01)
# lr = 0.05
# v = rbm.randomSample(imgs)
# h = np.zeros(100)
# w = np.random.normal(loc=0,scale=0.01,size=(h.size,v.size))

# for i in xrange(10000):
# 	print i
# 	v = rbm.randomSample(imgs)
# 	if i > 9995: 
# 		# plt.imshow(actH(v,h,w,stochastic=False).reshape(25,-1),cmap='gray')
# 		# plt.plot(actH(v,h,w,stochastic=False).flatten())
# 		# plt.show()
# 		for j in xrange(1000):
# 			print j
# 			h = rbm.actH(v,h,w)
# 			v = rbm.actV(v,h,w,stochastic=False)
# 		# plt.hist(w.flatten(),bins=100)
# 		# plt.show()
# 		img = v.reshape(100,100)
# 		plt.imshow(img,cmap='gray')
# 		plt.show()
# 	v = rbm.randomSample(imgs)
# 	w = rbm.updateWeights(v,h,w,lr,stochastic=True)

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
# X = []
# Y = []
# # for i in xrange(1,len(sigs)):
# for i in xrange(1,100):
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
# 	# corners[:,1] += x.min()
# 	# corners[:,0] += y.min()
# 	img01 = pilResize(imgO,(100,100))
# 	img02 = resize(imgO,(100,100))
# 	img01 = img01 > thresh
# 	img02 = img02 > thresh
# 	pathx = path[:,0]*img01.shape[1]
# 	pathy = path[:,1]*img01.shape[0]
# 	plt.subplot(1,2,1)
# 	plt.imshow(img01,cmap='gray')
# 	# plt.scatter(pathx,pathy,s=path[:,2])
# 	plt.subplot(1,2,2)
# 	plt.imshow(img02,cmap='gray')
# 	# plt.scatter(pathx,pathy,s=path[:,2])
# 	# plt.scatter(pathx,pathy,s=path[:,2])
# 	# plt.scatter(corners[:,1],corners[:,0],s=50,c='red')
# 	plt.show()
# 	startx = path[0][0]*img.shape[1]+x.min()
# 	starty = path[0][1]*img.shape[0]+y.min()
# 	start = np.transpose(np.array([[startx],[starty]]))
# 	corners[:,[0,1]] = corners[:,[1,0]]
# 	dists = cdist(start,corners)
# 	if dists.min() < 15:
# 		x,y = corners[dists.argmin()][0],corners[dists.argmin()][1]
# 		# corners = np.delete(corners,dists.argmin(),axis=0)
# 	else:
# 		x,y = int(startx),int(starty)
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
# 	wrong = Neighbors(imgO,wrongY-r/2,wrongX-r