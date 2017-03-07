from Tkinter import *
from tkMessageBox import *
import PIL.Image
from PIL import  ImageTk
import glob
import os
from shutil import copyfile

dataset_to_label = [img for img in glob.glob("../img/original_data/b_labeled/*jpg")]
output_folder = "../img/original_data/b_labeled2/"

for in_idx, img_path in enumerate(dataset_to_label):
	head, tail = os.path.split(img_path)
	name = tail.split(".")[0]
	extension = tail.split(".")[1]
	if "_showingteeth" in name:
		new_name = name.replace("_showingteeth", "")
		print(new_name)
		new_name = "_showingteeth"+new_name
		output_path = output_folder+"/"+new_name+"."+extension
		copyfile(img_path, output_path)
	else:
		new_name = name
		print(new_name)
		output_path = output_folder+"/"+new_name+"."+extension
		copyfile(img_path, output_path)
	