#Manual feature detector to slice mouth from the classified folders , this will be our true training data
import cv2
import numpy as np
import glob
import uuid
import caffe
import skimage.io
from util import histogram_equalization
from scipy.ndimage import zoom
from skimage.transform import resize

#note that here input_img_path will not have any label
#input full color image , outputs 50by50 mouth slice gray BGR2 1..255
def mouth_detect_single(input_img_path):
    face_cascade = cv2.CascadeClassifier('../lib/haarcascade/haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('../lib/haarcascade/mouth.xml')

    img_width = 50
    img_height = 50

    img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED) 
    #img = histogram_equalization(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    cv2.imwrite("../img/output_test_img/hmouthdetectsingle.jpg",gray_img)
    faces = face_cascade.detectMultiScale(gray_img, 3.4, 5)
    if(len(faces)>0):
        p = faces[0][0]
        q = faces[0][1]
        r = faces[0][2]
        s = faces[0][3]
        cv2.rectangle(img,(p,q),(p+r,q+s),(255,0,0),2)
        face_gray = gray_img[q:q+s, p:p+r]
        mouth = mouth_cascade.detectMultiScale(face_gray)
        biggestYIndex = -1
        biggestYPos = -1
        if len(mouth)==0:
                return
        for idx, val in enumerate(mouth):
            if val[1]>biggestYPos:
                biggestYPos=val[1]
                biggestYIndex = idx
        mp = mouth[biggestYIndex][0]
        mq = mouth[biggestYIndex][1]
        mr = mouth[biggestYIndex][2]
        ms = mouth[biggestYIndex][3]
        tt=cv2.rectangle(face_gray,(mp,mq),(mp+mr,mq+ms), (255,255,255),2)
        cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_region.jpg",tt)
        crop_img = face_gray[mq:mq+ms, mp:mp+mr]
        crop_img_resized = cv2.resize(crop_img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_rezized.jpg",crop_img_resized)
        return crop_img_resized 
    
#input full color image , outputs 50by50 mouth slice gray BGR2 1..255
def mouth_detect_bulk(input_folder,output_folder):
    #full color image input
    face_cascade = cv2.CascadeClassifier('../lib/haarcascade/haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('../lib/haarcascade/mouth.xml')
    transformed_data_set = [img for img in glob.glob(input_folder+"/*jpg")]

    img_width = 50
    img_height = 50

    for in_idx, img_path in enumerate(transformed_data_set):
        mouth = mouth_detect_single(img_path)
        if 'showingteeth' in img_path:
                guid = uuid.uuid4()
                uid_str = guid.urn
                str_guid = uid_str[9:]
                path = output_folder+"/"+str_guid+"_showingteeth.jpg"
                cv2.imwrite(path,mouth)
        else:
            guid = uuid.uuid4()
            uid_str = guid.urn
            str_guid = uid_str[9:]
            path = output_folder+"/"+str_guid+".jpg"
            cv2.imwrite(path,mouth)