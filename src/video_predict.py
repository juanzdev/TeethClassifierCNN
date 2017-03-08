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
IMAGE_WIDTH_MAIN = 480
IMAGE_HEIGHT_MAIN = 640

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
#CNN Definition
#extract mean data
mean_blob = caffe_pb2.BlobProto()
with open('../mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

mean_array = mean_array*0.003921568627

net = caffe.Net('../model/deploy.prototxt',1,weights='../model_snapshot/snap_fe_iter_2700.caffemodel')
net.blobs['data'].reshape(1,1, IMAGE_WIDTH, IMAGE_HEIGHT)  # image size is 227x227
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 0.00392156862745) 


def predict(image):
	#img = cv2.imread(individual_test_image, cv2.IMREAD_UNCHANGED)
	#img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	#img = histogram_equalization(img)
	img = image
	mouth_pre = mouth_detect_single(img) #mouth is grayscale 1..255 50x50 BGR
	if mouth_pre is not None:
		mouth_pre = mouth_pre[:,:,np.newaxis]
		mouth = transformer.preprocess('data', mouth_pre)
		net.blobs['data'].data[...] = mouth
		out = net.forward()
		pred = out['pred'].argmax()
		#print(individual_test_image)
		print("Prediction:")
		print(pred)
		print("Prediction probabilities")
		print(out['pred'])
		print(out['pred'].shape)
		if(pred == 1 and out['pred'][0][1]>0.9):
			return 1
		else:
			return 0



cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    result = predict(frame)
    if result == 1:
    	cv2.putText(frame, "Showing teeth",(50, 50),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,0,255))
    #print result
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")