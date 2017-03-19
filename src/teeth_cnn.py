import numpy as np
import sys
import caffe
import glob
import uuid
import cv2
from util import transform_img
from mouth_detector_dlib import mouth_detector
from caffe.proto import caffe_pb2
import os
import shutil
from util import histogram_equalization

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

class teeth_cnn:
	
	def __init__(self):
		self.init_net()

   	def init_net(self):
   		self.mean_blob = self.mean_blob_fn()
   		self.mean_array = np.asarray(self.mean_blob.data, dtype=np.float32).reshape((self.mean_blob.channels, self.mean_blob.height, self.mean_blob.width))
   		self.mean_array = self.mean_array*0.00390625
   		self.net = caffe.Net('../model/deploy.prototxt',1,weights='../model_snapshot/snap_fe_iter_8700.caffemodel')
   		self.net.blobs['data'].reshape(1,1, IMAGE_WIDTH, IMAGE_HEIGHT)
   		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
   		self.transformer.set_mean('data', self.mean_array)
   		self.transformer.set_transpose('data', (2,0,1))
   		self.transformer.set_raw_scale('data', 0.00390625)

	def mean_blob_fn(self):
		mean_blob = caffe_pb2.BlobProto()
		with open('../mean.binaryproto') as f:
			mean_blob.ParseFromString(f.read())

		return mean_blob


	def predict(self,image,mouth_detector):
		img = image
		mouth_pre,x,y,w,h = mouth_detector.mouth_detect_single(img,False)

		if mouth_pre is not None:
			mouth_pre = mouth_pre[:,:,np.newaxis]
			mouth = self.transformer.preprocess('data', mouth_pre)
			self.net.blobs['data'].data[...] = mouth
			out = self.net.forward()
			#pred = out['pred'].argmax()
			if(out['pred'][0][1]>0.55):
				return 1,out['pred'],x,y,w,h
			else:
				return 0,out['pred'],x,y,w,h
		else:
			return -1,0,0,0,0,0