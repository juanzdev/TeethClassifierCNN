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
		#pred = out['pred'].argmax()
		#print("Prediction:")
		#print(pred)
		print("Prediction probabilities")
		print(out['pred'])
		if(out['pred'][0][1]>0.70):
			return 1,out['pred'],x,y,w,h
		else:
			return 0,out['pred'],x,y,w,h
	else:
		return -1,0,0,0,0,0

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


size = cv2.getTextSize("Showing teeth", cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
x,y = (50,250)
label_top_left = (x - size[0]/2, y - size[1]/2)
mouth_detector_instance = mouth_detector()



# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
fourcc = cv2.cv.CV_FOURCC(*'mp4v')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')


cap = cv2.VideoCapture('../farewell-speech.mp4')
w = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH);
h = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT);
out = cv2.VideoWriter('output.avi',fourcc, 24.0, (int(w),int(h)))
#cap.set(3,500)
#cap.set(4,500)
#cap.set(5,30)

#cap = cv2.VideoCapture('../blackwhite.mp4')
cap.set(1,1);
ret, frame = cap.read()
while(cap.isOpened()):
	ret, frame = cap.read()
	copy_frame = frame.copy()
	
	result,prob,xf,yf,wf,hf = predict(copy_frame,mouth_detector_instance)
	#key = cv2.waitKey(20)
	
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	if result is not None:
	    if(result == 1):
	    	cv2.rectangle(frame, (xf,yf),(wf,hf),(0,255,0),1)
	    	prob_round = prob[0][1]
	    	print prob_round
	    	cv2.rectangle(frame, (wf,hf-int(((hf-yf)/1.3))),(wf+70,yf+int(((hf-yf)/1.3))),(0,255,0),-2)
	    	cv2.putText(frame, "Teeth!",(wf,hf-int(((hf-yf)/2))),cv2.FONT_HERSHEY_PLAIN,0.9,(0,0,0))
	    	cv2.putText(frame, str(prob_round),(wf,hf-int(((hf-yf)/2))+10),cv2.FONT_HERSHEY_PLAIN,0.9,(0,0,0))
	    	#out.write(frame)
	    	print "SHOWING TEETH!!!"
	    elif(result==0):
	    	cv2.rectangle(frame, (xf,yf),(wf,hf),(128,128,128),1)
	    	prob_round = prob[0][1]
	    	print prob_round
	    	cv2.rectangle(frame, (wf,hf-int(((hf-yf)/1.3))),(wf+70,yf+int(((hf-yf)/1.3))),(128,128,128),-2)
	    	cv2.putText(frame, "Teeth?",(wf,hf-int(((hf-yf)/2))),cv2.FONT_HERSHEY_PLAIN,0.9,(0,0,0))
	    	cv2.putText(frame, str(prob_round),(wf,hf-int(((hf-yf)/2))+10),cv2.FONT_HERSHEY_PLAIN,0.9,(0,0,0))
	    	
	    	print "SHOWING TEETH!!!"
	
	out.write(frame)
	cv2.imshow('frame',frame)
	
	if cv2.waitKey(80) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()
