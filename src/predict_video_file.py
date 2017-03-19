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
from teeth_cnn import teeth_cnn

mouth_detector_instance = mouth_detector()
teeth_cnn_instance = teeth_cnn()


size = cv2.getTextSize("Showing teeth", cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
x,y = (50,250)

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'mp4v')

cap = cv2.VideoCapture('../elon.mp4')
cap.set(1,19300);
ret, frame = cap.read()
#cv2.imshow('window_name', frame) # show frame on window
w = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH);
h = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT);
out = cv2.VideoWriter('output_elon.avi',fourcc, 24, (int(w),int(h)))
#cap.set(3,500)
#cap.set(4,500)
#cap.set(5,30)

ret, frame = cap.read()
while(cap.isOpened()):
	ret, frame = cap.read()
	copy_frame = frame.copy()
	result,prob,xf,yf,wf,hf = teeth_cnn_instance.predict(copy_frame,mouth_detector_instance)
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
	
	out.write(frame)
	cv2.imshow('frame',frame)
	
	if cv2.waitKey(200) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()
