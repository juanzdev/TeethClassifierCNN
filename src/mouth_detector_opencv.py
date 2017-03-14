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
import dlib
from project_face import frontalizer

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

class mouth_detector():
    def __init__(self):
        self.PATH_face_model = '../lib/shape_predictor_68_face_landmarks.dat'
        self.face_cascade = cv2.CascadeClassifier('../lib/haarcascade/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('../lib/haarcascade/haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier('../lib/haarcascade/mouth.xml')
        self.md_face = dlib.shape_predictor(self.PATH_face_model)
        self.fronter = frontalizer('../lib/ref3d.pkl')

    def mouth_detect_single(self,image,isPath):

        if isPath == True:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED) 
        else:
            img = image
        
        img = histogram_equalization(img)
        gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = self.face_cascade.detectMultiScale(gray_img1, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_gray = gray_img1[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if(len(eyes)>0):
                #valid face
                #height_region = (h + y)-y
                #width_region = (w + x) - x 
                #p = x + (width_region/4)
                #q = y + (height_region/2)
                #r = w - (width_region/2)
                #s = h - (height_region/2)

                p = x 
                q = y
                r = w
                s = h

                face_region = gray_img1[q:q+s, p:p+r]
                face_region_rect = dlib.rectangle(long(q),long(p),long(q+s),long(p+r))
                rectan = dlib.rectangle(long(x),long(y),long(x+w),long(y+h))
                shape = self.md_face(img,rectan)
                p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)
                rawfront, symfront = self.fronter.frontalization(img,face_region_rect,p2d)
                face_hog_mouth = symfront[165:220, 130:190]
                gray_img = cv2.cvtColor(face_hog_mouth, cv2.COLOR_BGR2GRAY) 
                crop_img_resized = cv2.resize(gray_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_rezized.jpg",gray_img)
                return crop_img_resized,rectan.left(),rectan.top(),rectan.right(),rectan.bottom()
        else:
            return None,-1,-1,-1,-1

    def mouth_detect_bulk(self,input_folder,output_folder):

        transformed_data_set = [img for img in glob.glob(input_folder+"/*jpg")]

        for in_idx, img_path in enumerate(transformed_data_set):
            mouth,x,y,w,h = self.mouth_detect_single(img_path,True)
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

    def negative_image(self,imagem):
        imagem = (255-imagem)
        return imagem

    def adaptative_threashold(self,input_img_path):
        img = cv2.imread(input_img_path,0)
        img = cv2.medianBlur(img,3)
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        #cv2.imwrite("../img/output_test_img/hmouthdetectsingle_adaptative.jpg",th3)
        return th3