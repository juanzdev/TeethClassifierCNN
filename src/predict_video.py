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
import math
from teeth_cnn import teeth_cnn

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
#vc.set(3,500)
#vc.set(4,500)
#vc.set(5,30)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

mouth_detector_instance = mouth_detector()
teeth_cnn_instance = teeth_cnn()

size = cv2.getTextSize("Showing teeth", cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
x,y = (50,250)

while rval:
	rval, frame = vc.read()
	copy_frame = frame.copy()
	result,prob,xf,yf,wf,hf = teeth_cnn_instance.predict(copy_frame,mouth_detector_instance)
	print prob
	if result is not None:
	    if(result == 1):
	    	cv2.rectangle(frame, (xf,yf),(wf,hf),(0,255,0),4,0)
	    	prob_round = prob[0][1]*100
	    	print prob_round
	    	cv2.rectangle(frame, (xf-2,yf-25),(wf+2,yf),(0,255,0),-1,0)
	    	cv2.rectangle(frame, (xf-2,hf),(xf+((wf-xf)/2),hf+25),(0,255,0),-1,0)
	    	cv2.putText(frame, "Teeth!!",(xf,hf+14),cv2.FONT_HERSHEY_PLAIN,1.2,0,2)
	    	cv2.putText(frame, str(prob_round)+"%",(xf,yf-10),cv2.FONT_HERSHEY_PLAIN,1.2,0,2)
	    	#out.write(frame)
	    	print "SHOWING TEETH!!!"
	    elif(result==0):
	    	cv2.rectangle(frame, (xf,yf),(wf,hf),(64,64,64),4,0)
	    	prob_round = prob[0][1]*100
	    	print prob_round
	    	cv2.rectangle(frame, (xf-2,yf-25),(wf+2,yf),(64,64,64),-1,0)
	    	cv2.rectangle(frame, (xf-2,hf),(xf+((wf-xf)/2),hf+25),(64,64,64),-1,0)
	    	cv2.putText(frame, "Teeth??",(xf,hf+14),cv2.FONT_HERSHEY_PLAIN,1.2,0,2)
	    	cv2.putText(frame, str(prob_round)+"%",(xf,yf-10),cv2.FONT_HERSHEY_PLAIN,1.2,0,2)

	cv2.imshow("preview", frame)
	cv2.waitKey(1)
cv2.destroyWindow("preview")