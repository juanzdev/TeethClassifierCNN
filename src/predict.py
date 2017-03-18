from __future__ import division
#Prediction file, this will use the trained ConvNet to predict data from a set of files on a folder or for a individual file 
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

BULK_PREDICTION = 0 #Set this to 0 to classify individual files

#if bulk prediction is set to 1 the net will predict all images on the configured path
test_set_folder_path = "../img/original_data/b_labeled"
#all the files will be moved to a showing teeth or not showing teeth folder on the test_output_result_folder_path path
test_output_result_folder_path = "../result" 
#if BULK_PREDICTION = 0 the net will classify only the file specified on individual_test_image
individual_test_image = "../ana.jpg"
#read all test images
original_data_set = [img for img in glob.glob(test_set_folder_path+"/*jpg")]

mouth_detector_instance = mouth_detector()
teeth_cnn_instance = teeth_cnn()

if BULK_PREDICTION==0:
	img = cv2.imread(individual_test_image, cv2.IMREAD_UNCHANGED)
	result,prob,xf,yf,wf,hf = teeth_cnn_instance.predict(img,mouth_detector_instance)
	print(individual_test_image)
	print("Prediction:")
	print(result)
	print("Prediction probabilities")
	print(prob)
else:
	files = glob.glob(test_output_result_folder_path+'/not_showing_teeth/*')
	for f in files:
		os.remove(f)

	files = glob.glob(test_output_result_folder_path+'/showing_teeth/*')
	for f in files:
		os.remove(f)
	
	#performance variables
	total_samples = 0
	total_positives_training = 0
	total_negatives_training = 0
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0
	
	for in_idx, img_path in enumerate(original_data_set):
		total_samples = total_samples + 1
		head, tail = os.path.split(img_path)
		img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		result,prob,xf,yf,wf,hf = teeth_cnn_instance.predict(img,mouth_detector_instance)
		print("Prediction:")
		print(result)
		print("Prediction probabilities")
		print(prob)
		if(result==1):
			if 'showingteeth' in tail:
				total_positives_training = total_positives_training + 1
				true_positive = true_positive + 1
			else:
				total_negatives_training = total_negatives_training + 1
				false_positive = false_positive + 1

			path = test_output_result_folder_path+"/showing_teeth/"+tail
			shutil.copy2(img_path, path)
		else:
			if 'showingteeth' in tail:
				total_positives_training = total_positives_training + 1
				false_negative = false_negative + 1
			else:
				total_negatives_training = total_negatives_training + 1
				true_negative = true_negative + 1

			path = test_output_result_folder_path+"/not_showing_teeth/"+tail
			shutil.copy2(img_path, path)

	print "Total samples %d" %total_samples
 	print "True positives %d" %true_positive
 	print "False positives %d" %false_positive
	print "True negative %d" %true_negative
	print "False negative %d" %false_negative
	
	accuracy = (true_negative + true_positive)/total_samples
	recall = true_positive / (true_positive + false_negative)
	precision = true_positive / (true_positive + false_positive)
	f1score = 2*((precision*recall)/(precision+recall))

	print "Accuracy  %.2f" %accuracy
	print "Recall  %.2f" %recall
	print "Precision  %.2f" %precision
	print "F1Score  %.2f" %f1score


