import csv
from skimage import draw
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import random
import cProfile

class TrainingImages:
	def __init__(self, csvFile = "../train.csv", imageDirectory = "../images/"):
		self.csvFile = csvFile
		self.imageDirectory = imageDirectory
		self.images = []

		
	def buildImages(self):
		with open(self.csvFile, 'rb') as strokeData:
			reader = csv.reader(strokeData, delimiter=",")
			for line in reader:
				self.appendCsvLine(line)
		self.addImageData()
			
	def appendCsvLine(self, line):
		'''Assumes the line is from a csv.reader object'''
		signatureId = line[1]
		if len(self.images) <= signatureId:
			newImage = Image(signatureId)
			self.images.append(newImage)
			newImage.append(line)
		else:
			self.images[(signatureId-1)].append(line)
				
	def addImageData(self):
		for image in self.images:
			print image.signatureId
		for image in self.images:
			print image.signatureId
			imageNumber = str(image.signatureId).zfill(4)
			filePath = self.imageDirectory + imageNumber + ".jpg"
			print filePath
			image.construct(filePath)

	def returnRandomImages(self, amount):
		return random.sample(self.images, amount)
	
class Image:
	def __init__(self, signatureId):
		self.imageData = None #list of lists for pixel values
		self.skeletonData = None
		self.timeData = [] #list of all time values, later normalized from 0 to 1
		self.strokeData = [] #list of x,y tuples correlating with timeData
		self.signatureId = signatureId
		
	def append(self, line):
		'''Assumes a csv reader line as input'''
		self.timeData.append(line[4])
		self.strokeData.append((line[5],line[6]))
		
	def construct(self, filePath):
		self.imageData = imread(filePath, as_grey=True)

	def toSkeleton(self):
		skeleton = skeletonize(self.imageData)
		self.skeletonData = skeleton

	def plot(self):
		plt.imgshow(self.image, cmap=plt.cm.gray)
		plt.axis('off')
		plt.show()

test = TrainingImages()
test.buildImages()
#images = test.returnRandomImages(1)
#images[0].plot()