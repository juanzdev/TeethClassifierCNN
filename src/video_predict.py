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

def predict(image,mouth_detector):
	img = image
	mouth_pre,x,y,w,h = mouth_detector.mouth_detect_single(img,False)

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
		if(pred == 1 and out['pred'][0][1]>0.7):
			return 1,x,y,w,h
		else:
			return 0,x,y,w,h
	else:
		return 0,0,0,0,0

#CNN Definition
#extract mean data
mean_blob = caffe_pb2.BlobProto()
with open('../mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

mean_array = mean_array*0.003921568627

net = caffe.Net('../model/deploy.prototxt',1,weights='../model_snapshot/snap_fe_iter_8700.caffemodel')
net.blobs['data'].reshape(1,1, IMAGE_WIDTH, IMAGE_HEIGHT) 
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 0.00392156862745) 



cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
#vc.set(3,500)
#vc.set(4,500)
#vc.set(5,30)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

size = cv2.getTextSize("Showing teeth", cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
x,y = (50,250)
label_top_left = (x - size[0]/2, y - size[1]/2)
mouth_detector_instance = mouth_detector()

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    result,xf,yf,wf,hf = predict(frame,mouth_detector_instance)
    print xf
    cv2.rectangle(frame, (xf,yf),(wf,hf),(0,255,0),1);
    if result is not None:
	    if(result == 1):
	    	cv2.rectangle(frame, (x,y),(x+size[0],y-size[1]),(0,255,0),-2);
	    	cv2.putText(frame, "Showing teeth",(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0))
    #print result
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")