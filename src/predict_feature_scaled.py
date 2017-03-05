#Predict file, this will use the trained ConvNet to predict data from a set of files on a folder or for a individual file 
import numpy as np
import sys
import caffe
import glob
import uuid
import cv2
from util import transform_img
from mouth_slicer import mouth_detect_single
from caffe.proto import caffe_pb2
import os
import shutil
from util import histogram_equalization

#CONFIGURATION VARIABLES

BULK_PREDICTION = 0 #Set this to 0 to classify individual files

#if bulk prediction is set to 1 the net will predict all images on the configured path
test_set_folder_path = "../img/original_data/b"
#all the files will be moved to a showing teeth or not showing teeth folder on the test_output_result_folder_path path
test_output_result_folder_path = "../result" 
#if BULK_PREDICTION = 0 the net will classify only the file specified on individual_test_image
individual_test_image = "../img/b.jpg"


#-------------------



#read all test images
original_data_set = [img for img in glob.glob(test_set_folder_path+"/*jpg")]
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227



#CNN Definition
#extract mean data
mean_blob = caffe_pb2.BlobProto()
with open('../mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

mean_array = mean_array*0.003921568627

net = caffe.Net('../model/deploy.prototxt',1,weights='../model_snapshot/snap_fe_iter_10000.caffemodel')
net.blobs['data'].reshape(1,1, 50, 50)  # image size is 227x227
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 0.00392156862745) 


if BULK_PREDICTION==0:
	img = cv2.imread(individual_test_image, cv2.IMREAD_UNCHANGED)
	#img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	#img = histogram_equalization(img)
	mouth_pre = mouth_detect_single(individual_test_image) #mouth is grayscale 1..255 50x50 BGR
	if mouth_pre is not None:
		mouth_pre = mouth_pre[:,:,np.newaxis]
		mouth = transformer.preprocess('data', mouth_pre)
		net.blobs['data'].data[...] = mouth
		out = net.forward()
		pred = out['pred'].argmax()
		print("Prediction:")
		print(pred)
		print("Prediction probabilities")
		print(out['pred'])
else:
	for in_idx, img_path in enumerate(original_data_set):
		head, tail = os.path.split(img_path)
		img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
		img = histogram_equalization(img)
		mouth_pre = mouth_detect_single(img_path)
		if mouth_pre is not None:
			mouth_pre = mouth_pre[:,:,np.newaxis]
			mouth = transformer.preprocess('data', mouth_pre)
			net.blobs['data'].data[...] = mouth
			out = net.forward()
			pred = out['pred'].argmax()
			print("Prediction:")
			print(pred)
			print("Prediction probabilities")
			print(out['pred'])
			if(pred==1):
				path = test_output_result_folder_path+"/showing_teeth/"+tail
				shutil.copy2(img_path, path)
			else:
				path = test_output_result_folder_path+"/not_showing_teeth/"+tail
				shutil.copy2(img_path, path)
			







