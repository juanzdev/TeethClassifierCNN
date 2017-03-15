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
#from project_face import project_face
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
        self.md_face = dlib.shape_predictor(self.PATH_face_model)
        self.fronter = frontalizer('../lib/ref3d.pkl')
        self.face_det = dlib.get_frontal_face_detector() #HOG

    def mouth_detect_single(self,image,isPath):

        if isPath == True:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED) 
        else:
            img = image
    
        img = histogram_equalization(img)
        #gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        facedets = self.face_det(img,1)
        #print facedets
        if len(facedets) > 0:
            facedet_obj= facedets[0]
            shape = self.md_face(img,facedet_obj)
            #print shape
            p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)
            rawfront, symfront = self.fronter.frontalization(img,facedet_obj,p2d)
            face_hog_mouth = symfront[165:220, 130:190] #get half-bottom part
            if(face_hog_mouth is not None):
                gray_img = cv2.cvtColor(face_hog_mouth, cv2.COLOR_BGR2GRAY) 
                #negative_mouth_region = self.negative_image(gray_img)
                crop_img_resized = cv2.resize(gray_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                #cv2.imwrite("../img/output_test_img/mouthdetectsingle_crop_rezized.jpg",crop_img_resized)
                return crop_img_resized,facedet_obj.left(),facedet_obj.top(),facedet_obj.right(),facedet_obj.bottom()
            else:
                return None,-1,-1,-1,-1
        else:
            return None,-1,-1,-1,-1

    def mouth_detect_bulk(self,input_folder,output_folder):

        transformed_data_set = [img for img in glob.glob(input_folder+"/*jpg")]

        for in_idx, img_path in enumerate(transformed_data_set):
            mouth = self.mouth_detect_single(img_path,True)
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