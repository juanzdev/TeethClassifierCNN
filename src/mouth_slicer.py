import cv2
import numpy as np
import glob
import uuid
import caffe
import skimage.io
from util import histogram_equalization
from scipy.ndimage import zoom
from skimage.transform import resize
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

def negative_image(imagem):
    imagem = (255-imagem)
    return imagem
    #cv2.imwrite(name, imagem)

def adaptative_threashold(input_img_path):
    img = cv2.imread(input_img_path,0)
    img = cv2.medianBlur(img,3)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    #cv2.imwrite("../img/output_test_img/hmouthdetectsingle_adaptative.jpg",th3)
    return th3


#note that here input_img_path will not have any label
#input full color image , outputs 50by50 mouth slice gray BGR2 1..255
def mouth_detect_single(input_img_path):
    face_cascade = cv2.CascadeClassifier('../lib/haarcascade/haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('../lib/haarcascade/mouth.xml')

    img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED) 
    
    #img = histogram_equalization(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    #value = 100
    #mask = (255 - gray_img) < value
    #grey_new = np.where((255 - gray_img) < value,255,gray_img+value)

    cv2.imwrite("../img/output_test_img/hmouthdetectsingle.jpg",gray_img)

    faces = face_cascade.detectMultiScale(gray_img, 1.6, 5)
    print(len(faces))
    if(len(faces)>0):
        height_region = (faces[0][3] + faces[0][1])-faces[0][1]
        width_region = (faces[0][2] + faces[0][0]) - faces[0][0] 
        p = faces[0][0] + (width_region/4)
        q = faces[0][1] + (height_region/2)
        r = faces[0][2] - (width_region/2)
        s = faces[0][3] - (height_region/2)
        #cv2.rectangle(gray_img,(p,q),(p+r,q+s),(255,0,0),2)
        mouth_region = gray_img[q:q+s, p:p+r]
        
        ##tt=cv2.rectangle(face_gray,(mp,mq),(mp+mr,mq+ms), (255,255,255),2)
        #cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_region.jpg",gray_img)
        negative_mouth_region = negative_image(mouth_region)
        crop_img = negative_mouth_region
        crop_img_resized = cv2.resize(crop_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
        #print mouth_region.shape
        #print crop_img_resized.shape
        #print crop_img.shape
        cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_rezized.jpg",crop_img_resized)
        cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_negative.jpg",negative_mouth_region)
        return crop_img_resized
    else:
        print "NOFACE"
        print input_img_path


#note that here input_img_path will not have any label
#input full color image , outputs 50by50 mouth slice gray BGR2 1..255
def mouth_detect_single(image):
    face_cascade = cv2.CascadeClassifier('../lib/haarcascade/haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('../lib/haarcascade/mouth.xml')

    img = image
    
    #img = histogram_equalization(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    #value = 100
    #mask = (255 - gray_img) < value
    #grey_new = np.where((255 - gray_img) < value,255,gray_img+value)

    #cv2.imwrite("../img/output_test_img/hmouthdetectsingle.jpg",gray_img)

    faces = face_cascade.detectMultiScale(gray_img, 1.6, 5)
    #print(len(faces))
    if(len(faces)>0):
        height_region = (faces[0][3] + faces[0][1])-faces[0][1]
        width_region = (faces[0][2] + faces[0][0]) - faces[0][0] 
        p = faces[0][0] + (width_region/4)
        q = faces[0][1] + (height_region/2)
        r = faces[0][2] - (width_region/2)
        s = faces[0][3] - (height_region/2)
        #cv2.rectangle(gray_img,(p,q),(p+r,q+s),(255,0,0),2)
        mouth_region = gray_img[q:q+s, p:p+r]
        
        ##tt=cv2.rectangle(face_gray,(mp,mq),(mp+mr,mq+ms), (255,255,255),2)
        #cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_region.jpg",gray_img)
        negative_mouth_region = negative_image(mouth_region)
        crop_img = negative_mouth_region
        crop_img_resized = cv2.resize(crop_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
        #print mouth_region.shape
        #print crop_img_resized.shape
        #print crop_img.shape
        #cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_rezized.jpg",crop_img_resized)
        #cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_negative.jpg",negative_mouth_region)
        return crop_img_resized
    else:
        print "NOFACE"
        #print input_img_path

#input full color image , outputs 50by50 mouth slice gray BGR2 1..255
def mouth_detect_bulk(input_folder,output_folder):
    #full color image input
    transformed_data_set = [img for img in glob.glob(input_folder+"/*jpg")]

    for in_idx, img_path in enumerate(transformed_data_set):
        mouth = mouth_detect_single(img_path)
        #print "DOS"
        #print mouth
        if 'showingteeth' in img_path:
            guid = uuid.uuid4()
            uid_str = guid.urn
            str_guid = uid_str[9:]
            path = output_folder+"/"+str_guid+"_showingteeth.jpg"
            #print(mouth)
            cv2.imwrite(path,mouth)
        else:
            guid = uuid.uuid4()
            uid_str = guid.urn
            str_guid = uid_str[9:]
            path = output_folder+"/"+str_guid+".jpg"
            #print(mouth)
            cv2.imwrite(path,mouth)